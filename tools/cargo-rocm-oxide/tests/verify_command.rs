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

impl Drop for TempWorkspace {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn cargo_rocm_oxide() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_cargo-rocm-oxide"))
}

fn run_cli(args: &[&str], cwd: &Path) -> Output {
    Command::new(cargo_rocm_oxide())
        .args(args)
        .current_dir(cwd)
        .output()
        .expect("failed to run cargo-rocm-oxide")
}

#[test]
fn help_lists_verify_command() {
    let output = run_cli(&["help"], Path::new(env!("CARGO_MANIFEST_DIR")));

    assert!(
        output.status.success(),
        "help command failed: stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("cargo rocm-oxide verify [--offline|--quick|--full]"),
        "help output did not list verify command:\n{stdout}"
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
