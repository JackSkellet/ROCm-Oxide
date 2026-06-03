use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

struct TempWorkspace {
    root: PathBuf,
}

struct TempProject {
    root: PathBuf,
}

impl TempWorkspace {
    fn new() -> Self {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after UNIX_EPOCH")
            .as_nanos();
        let root = env::temp_dir().join(format!(
            "rocm-oxide-cli-test-{}-{stamp}",
            std::process::id()
        ));
        fs::create_dir_all(root.join("tools/rocm-oxide-build"))
            .expect("failed to create fake tool manifest directory");
        fs::create_dir_all(root.join("scripts")).expect("failed to create fake scripts directory");
        fs::create_dir_all(root.join("nested/workdir"))
            .expect("failed to create fake nested workdir");
        fs::write(
            root.join("tools/rocm-oxide-build/Cargo.toml"),
            "[package]\nname = \"rocm-oxide-build\"\nversion = \"0.0.0\"\nedition = \"2024\"\n",
        )
        .expect("failed to write fake tool manifest");
        Self { root }
    }

    fn install_verify_script(&self) -> PathBuf {
        let script = self.root.join("scripts/verify.sh");
        fs::write(
            &script,
            "#!/usr/bin/env bash\nset -euo pipefail\nprintf '%s\\n' \"$PWD\" > \"$ROCM_OXIDE_VERIFY_LOG\"\nprintf '%s\\n' \"$@\" >> \"$ROCM_OXIDE_VERIFY_LOG\"\n",
        )
        .expect("failed to write fake verify script");

        #[cfg(unix)]
        {
            let mut permissions = fs::metadata(&script)
                .expect("failed to stat fake verify script")
                .permissions();
            permissions.set_mode(0o755);
            fs::set_permissions(&script, permissions)
                .expect("failed to make fake verify script executable");
        }

        script
    }

    fn nested_workdir(&self) -> PathBuf {
        self.root.join("nested/workdir")
    }
}

impl TempProject {
    fn new() -> Self {
        let root = temp_root("rocm-oxide-cli-consumer-test");
        fs::create_dir_all(root.join("nested/workdir"))
            .expect("failed to create fake nested workdir");
        fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"consumer\"\nversion = \"0.0.0\"\nedition = \"2024\"\n",
        )
        .expect("failed to write consumer manifest");
        Self { root }
    }

    fn nested_workdir(&self) -> PathBuf {
        self.root.join("nested/workdir")
    }
}

impl Drop for TempProject {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

impl Drop for TempWorkspace {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn cargo_rocm_oxide() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_cargo-rocm-oxide"))
}

fn temp_root(prefix: &str) -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should be after UNIX_EPOCH")
        .as_nanos();
    env::temp_dir().join(format!("{prefix}-{}-{stamp}", std::process::id()))
}

fn write_executable(path: &Path, contents: &str) {
    fs::write(path, contents).expect("failed to write fake executable");
    #[cfg(unix)]
    {
        let mut permissions = fs::metadata(path)
            .expect("failed to stat fake executable")
            .permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(path, permissions).expect("failed to make fake executable executable");
    }
}

fn run_cli(args: &[&str], cwd: &Path) -> Output {
    Command::new(cargo_rocm_oxide())
        .args(args)
        .current_dir(cwd)
        .output()
        .expect("failed to run cargo-rocm-oxide")
}

#[test]
fn help_lists_pipeline_debug_and_verify_commands() {
    let output = run_cli(&["help"], Path::new(env!("CARGO_MANIFEST_DIR")));

    assert!(
        output.status.success(),
        "help command failed: stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("cargo rocm-oxide verify [--host-ci|--offline|--quick|--full]"),
        "help output did not list verify command:\n{stdout}"
    );
    assert!(
        stdout.contains("cargo rocm-oxide pipeline [--build] [--crate PATH] [--output-stem NAME]"),
        "help output did not list pipeline options:\n{stdout}"
    );
    assert!(
        stdout.contains("cargo rocm-oxide debug [cargo-run-args]"),
        "help output did not list debug command:\n{stdout}"
    );
}

#[test]
fn verify_routes_to_workspace_script_from_nested_directory() {
    let workspace = TempWorkspace::new();
    workspace.install_verify_script();
    let log = workspace.root.join("verify.log");

    let output = Command::new(cargo_rocm_oxide())
        .args(["rocm-oxide", "verify", "--quick"])
        .current_dir(workspace.nested_workdir())
        .env("ROCM_OXIDE_VERIFY_LOG", &log)
        .output()
        .expect("failed to run cargo-rocm-oxide verify");

    assert!(
        output.status.success(),
        "verify command failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let log = fs::read_to_string(log).expect("verify script did not write its log");
    let lines = log.lines().collect::<Vec<_>>();
    assert_eq!(lines.first(), Some(&workspace.root.to_string_lossy().as_ref()));
    assert_eq!(lines.get(1), Some(&"--quick"));
    assert_eq!(lines.len(), 2, "unexpected extra verify arguments: {lines:?}");
}

#[test]
fn pipeline_routes_to_external_build_tool_from_consumer_project() {
    let project = TempProject::new();
    let metadata = project.root.join(
        "device/target/amdgcn-amd-amdhsa/release/consumer_device.metadata.json",
    );
    fs::create_dir_all(metadata.parent().expect("metadata should have parent"))
        .expect("failed to create metadata directory");
    fs::write(&metadata, "{}").expect("failed to write fake metadata");

    let fake_build = project.root.join("fake-rocm-oxide-build");
    let log = project.root.join("build.log");
    write_executable(
        &fake_build,
        "#!/usr/bin/env bash\nprintf 'cwd=%s\\n' \"$PWD\" >> \"$ROCM_OXIDE_BUILD_LOG\"\nprintf 'args=' >> \"$ROCM_OXIDE_BUILD_LOG\"\nprintf '[%s]' \"$@\" >> \"$ROCM_OXIDE_BUILD_LOG\"\nprintf '\\n' >> \"$ROCM_OXIDE_BUILD_LOG\"\n",
    );

    let output = Command::new(cargo_rocm_oxide())
        .args([
            "rocm-oxide",
            "pipeline",
            "--build",
            "--crate",
            "device",
            "--output-stem",
            "consumer_device",
        ])
        .current_dir(project.nested_workdir())
        .env("ROCM_OXIDE_BUILD", &fake_build)
        .env("ROCM_OXIDE_BUILD_LOG", &log)
        .output()
        .expect("failed to run cargo-rocm-oxide pipeline");

    assert!(
        output.status.success(),
        "pipeline command failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let log = fs::read_to_string(log).expect("fake build tool did not write log");
    let root = project.root.to_string_lossy();
    assert!(
        log.contains(&format!(
            "cwd={root}\nargs=[--crate][device][--output-stem][consumer_device]"
        )),
        "build invocation was not rooted/forwarded correctly:\n{log}"
    );
    assert!(
        log.contains(&format!(
            "cwd={root}\nargs=[--inspect-metadata][{}]",
            metadata.display()
        )),
        "inspect invocation was not rooted/forwarded correctly:\n{log}"
    );
}

#[test]
fn debug_runs_cargo_from_consumer_project_with_device_debug_enabled() {
    let project = TempProject::new();
    let fake_cargo = project.root.join("fake-cargo");
    let log = project.root.join("cargo.log");
    write_executable(
        &fake_cargo,
        "#!/usr/bin/env bash\nprintf 'cwd=%s\\n' \"$PWD\" > \"$ROCM_OXIDE_CARGO_LOG\"\nprintf 'debug=%s\\n' \"${ROCM_OXIDE_DEVICE_DEBUG:-}\" >> \"$ROCM_OXIDE_CARGO_LOG\"\nprintf 'args=' >> \"$ROCM_OXIDE_CARGO_LOG\"\nprintf '[%s]' \"$@\" >> \"$ROCM_OXIDE_CARGO_LOG\"\nprintf '\\n' >> \"$ROCM_OXIDE_CARGO_LOG\"\n",
    );

    let output = Command::new(cargo_rocm_oxide())
        .args([
            "rocm-oxide",
            "debug",
            "--example",
            "smoke",
            "--",
            "--case",
            "one",
        ])
        .current_dir(project.nested_workdir())
        .env("CARGO", &fake_cargo)
        .env("ROCM_OXIDE_CARGO_LOG", &log)
        .output()
        .expect("failed to run cargo-rocm-oxide debug");

    assert!(
        output.status.success(),
        "debug command failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let log = fs::read_to_string(log).expect("fake cargo did not write log");
    assert_eq!(
        log,
        format!(
            "cwd={}\ndebug=1\nargs=[run][--example][smoke][--][--case][one]\n",
            project.root.display()
        )
    );
}

#[test]
fn new_project_scaffold_allows_default_pipeline() {
    let temp = temp_root("rocm-oxide-new-project-test");
    let app = temp.join("app");
    fs::create_dir_all(&temp).expect("failed to create temp parent");
    let output = run_cli(&["rocm-oxide", "new", app.to_str().unwrap()], Path::new(env!("CARGO_MANIFEST_DIR")));
    assert!(
        output.status.success(),
        "new command failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(app.join("device-spike/Cargo.toml").is_file());
    assert!(app.join("device-spike/src/lib.rs").is_file());
    assert!(app.join("build.rs").is_file());
    let build_rs = fs::read_to_string(app.join("build.rs")).expect("build script should be readable");
    assert!(build_rs.contains("rocm-oxide-build"));
    assert!(build_rs.contains("ROCM_OXIDE_DEVICE_MANIFEST"));
    assert!(build_rs.contains("--output-stem"));
    let host_main =
        fs::read_to_string(app.join("src/main.rs")).expect("host main should be readable");
    assert!(host_main.contains("include!(env!(\"ROCM_OXIDE_DEVICE_BINDINGS\"))"));
    assert!(host_main.contains("DeviceKernels::load_embedded"));
    fs::create_dir_all(app.join("src/nested")).expect("failed to create nested app directory");

    let fake_build = app.join("fake-rocm-oxide-build");
    let log = app.join("build.log");
    write_executable(
        &fake_build,
        "#!/usr/bin/env bash\nprintf 'cwd=%s\\n' \"$PWD\" >> \"$ROCM_OXIDE_BUILD_LOG\"\nprintf 'args=' >> \"$ROCM_OXIDE_BUILD_LOG\"\nprintf '[%s]' \"$@\" >> \"$ROCM_OXIDE_BUILD_LOG\"\nprintf '\\n' >> \"$ROCM_OXIDE_BUILD_LOG\"\n",
    );
    let output = Command::new(cargo_rocm_oxide())
        .args(["rocm-oxide", "pipeline", "--build"])
        .current_dir(app.join("src/nested"))
        .env("ROCM_OXIDE_BUILD", &fake_build)
        .env("ROCM_OXIDE_BUILD_LOG", &log)
        .output()
        .expect("failed to run default scaffold pipeline");
    assert!(
        output.status.success(),
        "pipeline command failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let log = fs::read_to_string(log).expect("fake build tool did not write log");
    assert_eq!(
        log,
        format!(
            "cwd={}\nargs=[--crate][device-spike][--output-stem][rocm_oxide_device_spike]\n",
            app.display()
        )
    );
    let _ = fs::remove_dir_all(temp);
}
