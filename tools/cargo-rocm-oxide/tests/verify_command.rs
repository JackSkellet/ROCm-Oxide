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
        fs::create_dir_all(&root).expect("failed to create fake workspace root");
        fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"rocm-oxide\"\nversion = \"0.0.0\"\nedition = \"2024\"\n",
        )
        .expect("failed to write fake runtime manifest");
        fs::create_dir_all(root.join("crates/rocm-oxide-device"))
            .expect("failed to create fake device crate");
        fs::write(
            root.join("crates/rocm-oxide-device/Cargo.toml"),
            "[package]\nname = \"rocm-oxide-device\"\nversion = \"0.0.0\"\nedition = \"2024\"\n",
        )
        .expect("failed to write fake device manifest");
        fs::create_dir_all(root.join("crates/rocm-oxide-kernel"))
            .expect("failed to create fake kernel crate");
        fs::write(
            root.join("crates/rocm-oxide-kernel/Cargo.toml"),
            "[package]\nname = \"rocm-oxide-kernel\"\nversion = \"0.0.0\"\nedition = \"2024\"\n",
        )
        .expect("failed to write fake kernel manifest");
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
    assert!(
        stdout.contains("cargo rocm-oxide new <path> --local ROCM_OXIDE_WORKSPACE"),
        "help output did not list new project options:\n{stdout}"
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
    assert_eq!(
        lines.first(),
        Some(&workspace.root.to_string_lossy().as_ref())
    );
    assert_eq!(lines.get(1), Some(&"--quick"));
    assert_eq!(
        lines.len(),
        2,
        "unexpected extra verify arguments: {lines:?}"
    );
}

#[test]
fn doctor_forwards_report_flags_to_build_tool() {
    let project = TempProject::new();
    let fake_build = project.root.join("fake-rocm-oxide-build");
    let log = project.root.join("build.log");
    write_executable(
        &fake_build,
        "#!/usr/bin/env bash\nprintf 'cwd=%s\\n' \"$PWD\" > \"$ROCM_OXIDE_BUILD_LOG\"\nprintf 'args=' >> \"$ROCM_OXIDE_BUILD_LOG\"\nprintf '[%s]' \"$@\" >> \"$ROCM_OXIDE_BUILD_LOG\"\nprintf '\\n' >> \"$ROCM_OXIDE_BUILD_LOG\"\n",
    );

    let output = Command::new(cargo_rocm_oxide())
        .args(["rocm-oxide", "doctor", "--json"])
        .current_dir(project.nested_workdir())
        .env("ROCM_OXIDE_BUILD", &fake_build)
        .env("ROCM_OXIDE_BUILD_LOG", &log)
        .output()
        .expect("failed to run cargo-rocm-oxide doctor");
    assert!(
        output.status.success(),
        "doctor command failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let log = fs::read_to_string(log).expect("fake build tool did not write log");
    assert_eq!(
        log,
        format!("cwd={}\nargs=[--doctor][--json]\n", project.root.display())
    );
}

#[test]
fn pipeline_routes_to_external_build_tool_from_consumer_project() {
    let project = TempProject::new();
    let metadata = project
        .root
        .join("device/target/amdgcn-amd-amdhsa/release/consumer_device.metadata.json");
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
    let output = run_cli(
        &["rocm-oxide", "new", app.to_str().unwrap()],
        Path::new(env!("CARGO_MANIFEST_DIR")),
    );
    assert!(
        output.status.success(),
        "new command failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(app.join("device-spike/Cargo.toml").is_file());
    assert!(app.join("device-spike/src/lib.rs").is_file());
    let device_source = fs::read_to_string(app.join("device-spike/src/lib.rs"))
        .expect("generated device source should be readable");
    assert!(
        device_source.contains("for_each_element(n, |i|"),
        "generated starter should use the ergonomic element helper:\n{device_source}"
    );
    assert!(
        device_source.contains("out.write(i, i.as_usize() as u32);"),
        "generated starter should use bounded DeviceSliceMut::write:\n{device_source}"
    );
    assert!(app.join(".vscode/settings.json").is_file());
    assert!(app.join(".vscode/tasks.json").is_file());
    assert!(app.join(".vscode/extensions.json").is_file());
    assert!(app.join(".vscode/rocm-oxide.code-snippets").is_file());
    let editor_settings =
        fs::read_to_string(app.join(".vscode/settings.json")).expect("editor settings");
    assert!(
        editor_settings.contains("\"device-spike/Cargo.toml\""),
        "editor settings should link the generated device crate:\n{editor_settings}"
    );
    assert!(
        editor_settings.contains("\"rust-analyzer.checkOnSave\": false"),
        "editor settings should avoid host-target check-on-save for AMDGPU device code:\n{editor_settings}"
    );
    let editor_tasks = fs::read_to_string(app.join(".vscode/tasks.json")).expect("editor tasks");
    assert!(
        editor_tasks.contains("ROCm-Oxide: check scaffold"),
        "editor tasks should include scaffold validation:\n{editor_tasks}"
    );
    assert!(
        editor_tasks.contains("cargo build") && editor_tasks.contains("cargo run"),
        "editor tasks should include build and run commands:\n{editor_tasks}"
    );
    let snippets =
        fs::read_to_string(app.join(".vscode/rocm-oxide.code-snippets")).expect("snippets");
    assert!(
        snippets.contains("rocm-kernel-1d") && snippets.contains("rocm-vector-add"),
        "editor snippets should include ROCm-Oxide kernel shortcuts:\n{snippets}"
    );
    assert!(app.join("build.rs").is_file());
    let build_rs =
        fs::read_to_string(app.join("build.rs")).expect("build script should be readable");
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

#[test]
fn new_project_accepts_explicit_workspace_path() {
    let workspace = TempWorkspace::new();
    let temp = temp_root("rocm-oxide-new-local-project-test");
    let app = temp.join("app");
    fs::create_dir_all(&temp).expect("failed to create temp parent");

    let output = Command::new(cargo_rocm_oxide())
        .args([
            "rocm-oxide",
            "new",
            "--path",
            workspace
                .root
                .to_str()
                .expect("workspace path should be utf-8"),
            app.to_str().expect("app path should be utf-8"),
        ])
        .current_dir(&temp)
        .output()
        .expect("failed to run cargo-rocm-oxide new --path");
    assert!(
        output.status.success(),
        "new --path command failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let manifest = fs::read_to_string(app.join("Cargo.toml")).expect("manifest should be readable");
    assert!(
        manifest.contains("rocm-oxide = { path = "),
        "generated manifest should use a local path dependency:\n{manifest}"
    );
    assert!(
        !manifest.contains(&workspace.root.to_string_lossy().to_string()),
        "generated manifest should not embed an absolute workspace path:\n{manifest}"
    );

    let output = Command::new(cargo_rocm_oxide())
        .args(["rocm-oxide", "check-consumer"])
        .current_dir(&app)
        .output()
        .expect("failed to run check-consumer");
    assert!(
        output.status.success(),
        "check-consumer failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("all checks passed"),
        "check-consumer did not pass generated --path scaffold:\n{stdout}"
    );

    let _ = fs::remove_dir_all(temp);
}

#[test]
fn new_project_accepts_git_dependency_scaffold() {
    let temp = temp_root("rocm-oxide-new-git-project-test");
    let app = temp.join("app");
    fs::create_dir_all(&temp).expect("failed to create temp parent");

    let output = Command::new(cargo_rocm_oxide())
        .args([
            "rocm-oxide",
            "new",
            app.to_str().expect("app path should be utf-8"),
            "--git",
            "https://github.com/JackSkellet/ROCm-Oxide",
            "--rev",
            "abc1234",
        ])
        .current_dir(&temp)
        .output()
        .expect("failed to run cargo-rocm-oxide new --git");
    assert!(
        output.status.success(),
        "new --git command failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let manifest = fs::read_to_string(app.join("Cargo.toml")).expect("manifest should be readable");
    assert!(
        manifest.contains(
            r#"rocm-oxide = { git = "https://github.com/JackSkellet/ROCm-Oxide", rev = "abc1234" }"#
        ),
        "generated host manifest should use the requested git dependency:\n{manifest}"
    );
    assert!(
        !manifest.contains("path = "),
        "git scaffold host manifest should not contain path dependencies:\n{manifest}"
    );

    let device_manifest =
        fs::read_to_string(app.join("device-spike/Cargo.toml")).expect("device manifest");
    assert!(
        device_manifest.contains(
            r#"rocm-oxide-device = { git = "https://github.com/JackSkellet/ROCm-Oxide", rev = "abc1234" }"#
        ),
        "generated device manifest should use git for device API:\n{device_manifest}"
    );
    assert!(
        device_manifest.contains(
            r#"rocm-oxide-kernel = { git = "https://github.com/JackSkellet/ROCm-Oxide", rev = "abc1234" }"#
        ),
        "generated device manifest should use git for kernel macro:\n{device_manifest}"
    );

    let build_rs = fs::read_to_string(app.join("build.rs")).expect("build.rs should be readable");
    assert!(
        build_rs.contains("const RUNTIME_PATH: Option<&str> = None;"),
        "git scaffold build.rs should not embed a runtime path:\n{build_rs}"
    );
    assert!(
        build_rs.contains("cargo install --git"),
        "git scaffold build.rs should include an install hint:\n{build_rs}"
    );

    let output = Command::new(cargo_rocm_oxide())
        .args(["rocm-oxide", "check-consumer"])
        .current_dir(&app)
        .output()
        .expect("failed to run check-consumer");
    assert!(
        output.status.success(),
        "check-consumer failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("all checks passed"),
        "check-consumer did not pass generated --git scaffold:\n{stdout}"
    );

    let _ = fs::remove_dir_all(temp);
}

#[test]
fn new_project_rejects_standalone_until_artifacts_exist() {
    let temp = temp_root("rocm-oxide-new-standalone-project-test");
    let app = temp.join("app");
    fs::create_dir_all(&temp).expect("failed to create temp parent");

    let output = run_cli(
        &[
            "rocm-oxide",
            "new",
            app.to_str().expect("app path should be utf-8"),
            "--standalone",
        ],
        Path::new(env!("CARGO_MANIFEST_DIR")),
    );
    assert!(
        !output.status.success(),
        "standalone command unexpectedly succeeded: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("new --standalone` is not supported yet"),
        "standalone error was not actionable:\n{stderr}"
    );
    assert!(
        !app.exists(),
        "standalone failure should not create the project"
    );

    let _ = fs::remove_dir_all(temp);
}
