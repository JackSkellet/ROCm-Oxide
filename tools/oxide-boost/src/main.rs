mod gui;

use std::env;
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = env::args_os().skip(1).collect::<Vec<_>>();
    let Some(command) = args.first().and_then(|arg| arg.to_str()) else {
        print_help();
        return Ok(());
    };

    match command {
        "doctor" => doctor(),
        "analyze" => analyze(&args[1..]),
        "deep-analyze" => deep_analyze(&args[1..]),
        "gui" => gui::run(),
        "run" => run_profile(&args[1..]),
        "patch" => patch(&args[1..]),
        "report" => report(),
        "help" | "--help" | "-h" => {
            print_help();
            Ok(())
        }
        other => Err(format!("unknown oxide-boost command `{other}`")),
    }
}

fn print_help() {
    println!(
        "Usage:
    oxide-boost doctor
    oxide-boost analyze [--edge] <game-dir>
    oxide-boost deep-analyze [--ghidra] <game-dir>
    oxide-boost gui
    oxide-boost run --profile <name> -- <program> [args...]
    oxide-boost patch apply --profile <name> --game-dir <dir> --target <relative-file> --modified <file>
    oxide-boost patch restore --profile <name> --game-dir <dir> --target <relative-file>
    oxide-boost patch status [--profile <name>]
    oxide-boost report

This is not an upscaler. It launches the chosen program with isolated persistent
shader/cache directories, can apply reversible per-game file patches, and records
run/cache behavior for repeat-run tuning."
    );
}

#[derive(Debug, Clone)]
struct RunArgs {
    profile: String,
    command: Vec<OsString>,
}

impl RunArgs {
    fn parse(args: &[OsString]) -> Result<Self, String> {
        let mut profile = None;
        let mut index = 0;
        while index < args.len() {
            let arg = &args[index];
            if arg == "--" {
                index += 1;
                break;
            }
            if arg == "--profile" {
                index += 1;
                profile = args
                    .get(index)
                    .and_then(|value| value.to_str())
                    .map(str::to_owned);
            } else if let Some(value) = arg.to_str().and_then(|arg| arg.strip_prefix("--profile="))
            {
                profile = Some(value.to_string());
            } else {
                return Err(format!("unknown run argument `{}`", arg.to_string_lossy()));
            }
            index += 1;
        }

        let command = args[index..].to_vec();
        if command.is_empty() {
            return Err("oxide-boost run requires `-- <program> [args...]`".to_string());
        }
        let profile = profile.unwrap_or_else(|| infer_profile_name(&command[0]));
        Ok(Self {
            profile: sanitize_profile_name(&profile),
            command,
        })
    }
}

fn doctor() -> Result<(), String> {
    let root = boost_root()?;
    let paths = BoostPaths::new(&root, "doctor");
    println!("Oxide Boost doctor");
    println!("root: {}", root.display());
    println!("mesa shader cache env: {}", paths.mesa_cache.display());
    println!("dxvk state cache env: {}", paths.dxvk_cache.display());
    println!("run logs: {}", root.join("runs").display());

    report_program("vulkaninfo");
    report_program("glxinfo");
    report_program("wine");
    report_program("gamescope");
    if let Some(path) = gui::ghidra_headless_path() {
        println!("ghidra analyzeHeadless: {}", path.display());
    } else {
        println!("ghidra analyzeHeadless: not found");
    }
    println!("ok: launcher is ready; missing optional tools only limit diagnostics");
    Ok(())
}

fn report_program(program: &str) {
    let found = env::var_os("PATH")
        .and_then(|path| find_in_path(program, &path))
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "not found".to_string());
    println!("{program}: {found}");
}

fn analyze(args: &[OsString]) -> Result<(), String> {
    let edge = args.first().is_some_and(|arg| arg == "--edge");
    let game_dir_arg = if edge { args.get(1) } else { args.first() };
    let Some(game_dir) = game_dir_arg else {
        return Err("oxide-boost analyze requires a game directory".to_string());
    };
    let report = if edge {
        gui::analyze_game_dir_edge_for_cli(PathBuf::from(game_dir))?
    } else {
        gui::analyze_game_dir_for_cli(PathBuf::from(game_dir))?
    };
    println!("{report}");
    Ok(())
}

fn deep_analyze(args: &[OsString]) -> Result<(), String> {
    let mut run_ghidra = false;
    let mut game_dir = None;
    for arg in args {
        if arg == "--ghidra" {
            run_ghidra = true;
        } else if game_dir.is_none() {
            game_dir = Some(arg);
        } else {
            return Err(format!(
                "unknown deep-analyze argument `{}`",
                arg.to_string_lossy()
            ));
        }
    }
    let Some(game_dir) = game_dir else {
        return Err("oxide-boost deep-analyze requires a game directory".to_string());
    };
    let report = gui::deep_analyze_game_dir_for_cli(PathBuf::from(game_dir), run_ghidra)?;
    println!("{report}");
    Ok(())
}

fn find_in_path(program: &str, path: &OsString) -> Option<PathBuf> {
    env::split_paths(path)
        .map(|dir| dir.join(program))
        .find(|path| path.is_file())
}

fn run_profile(args: &[OsString]) -> Result<(), String> {
    let args = RunArgs::parse(args)?;
    let root = boost_root()?;
    let paths = BoostPaths::new(&root, &args.profile);
    paths.create().map_err(|err| {
        format!(
            "failed to create cache directories for `{}`: {err}",
            args.profile
        )
    })?;

    let before = CacheSnapshot::read(&paths)
        .map_err(|err| format!("failed to read initial cache state: {err}"))?;
    let started_ms = unix_ms();
    let start = Instant::now();
    let status = Command::new(&args.command[0])
        .args(&args.command[1..])
        .env("MESA_SHADER_CACHE_DISABLE", "false")
        .env("MESA_SHADER_CACHE_DIR", &paths.mesa_cache)
        .env("MESA_SHADER_CACHE_MAX_SIZE", "20G")
        .env("DXVK_STATE_CACHE", "1")
        .env("DXVK_STATE_CACHE_PATH", &paths.dxvk_cache)
        .env("ROCM_OXIDE_BOOST_PROFILE", &args.profile)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .map_err(|err| {
            format!(
                "failed to launch `{}`: {err}",
                args.command[0].to_string_lossy()
            )
        })?;
    let elapsed_ms = start.elapsed().as_millis() as u64;
    let after = CacheSnapshot::read(&paths)
        .map_err(|err| format!("failed to read final cache state: {err}"))?;
    let run = RunRecord {
        profile: args.profile,
        command: args
            .command
            .iter()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect(),
        started_ms,
        elapsed_ms,
        status_code: status.code(),
        before,
        after,
    };
    run.write(&root)
        .map_err(|err| format!("failed to write run record: {err}"))?;

    println!(
        "oxide-boost: profile `{}` finished in {:.2}s, mesa cache {}, dxvk cache {}",
        run.profile,
        elapsed_ms as f64 / 1000.0,
        format_bytes(run.after.mesa_bytes),
        format_bytes(run.after.dxvk_bytes),
    );

    match status.code() {
        Some(code) => std::process::exit(code),
        None => std::process::exit(1),
    }
}

fn report() -> Result<(), String> {
    let root = boost_root()?;
    let profiles = root.join("profiles");
    println!("Oxide Boost report");
    println!("root: {}", root.display());
    if !profiles.is_dir() {
        println!("no profiles yet");
        return Ok(());
    }

    println!(
        "{:<32} {:>12} {:>12} {:>12}",
        "profile", "mesa", "dxvk", "total"
    );
    for entry in fs::read_dir(&profiles).map_err(|err| format!("failed to read profiles: {err}"))? {
        let entry = entry.map_err(|err| format!("failed to read profile entry: {err}"))?;
        if !entry.path().is_dir() {
            continue;
        }
        let profile = entry.file_name().to_string_lossy().into_owned();
        let paths = BoostPaths::new(&root, &profile);
        let snapshot = CacheSnapshot::read(&paths)
            .map_err(|err| format!("failed to inspect cache for profile `{profile}`: {err}"))?;
        println!(
            "{:<32} {:>12} {:>12} {:>12}",
            profile,
            format_bytes(snapshot.mesa_bytes),
            format_bytes(snapshot.dxvk_bytes),
            format_bytes(snapshot.total_bytes()),
        );
    }
    Ok(())
}

fn patch(args: &[OsString]) -> Result<(), String> {
    let Some(command) = args.first().and_then(|arg| arg.to_str()) else {
        return Err("oxide-boost patch requires apply, restore, or status".to_string());
    };

    match command {
        "apply" => patch_apply(&args[1..]),
        "restore" => patch_restore(&args[1..]),
        "status" => patch_status(&args[1..]),
        other => Err(format!("unknown oxide-boost patch command `{other}`")),
    }
}

#[derive(Debug)]
struct PatchArgs {
    profile: String,
    game_dir: PathBuf,
    target: PathBuf,
    modified: Option<PathBuf>,
    dry_run: bool,
}

impl PatchArgs {
    fn parse(args: &[OsString], needs_modified: bool) -> Result<Self, String> {
        let mut profile = None;
        let mut game_dir = None;
        let mut target = None;
        let mut modified = None;
        let mut dry_run = false;
        let mut index = 0;

        while index < args.len() {
            let arg = &args[index];
            if arg == "--profile" {
                index += 1;
                profile = args
                    .get(index)
                    .and_then(|value| value.to_str())
                    .map(str::to_owned);
            } else if arg == "--game-dir" {
                index += 1;
                game_dir = args.get(index).map(PathBuf::from);
            } else if arg == "--target" {
                index += 1;
                target = args.get(index).map(PathBuf::from);
            } else if arg == "--modified" {
                index += 1;
                modified = args.get(index).map(PathBuf::from);
            } else if arg == "--dry-run" {
                dry_run = true;
            } else if let Some(value) = arg.to_str().and_then(|arg| arg.strip_prefix("--profile="))
            {
                profile = Some(value.to_string());
            } else if let Some(value) = arg.to_str().and_then(|arg| arg.strip_prefix("--game-dir="))
            {
                game_dir = Some(PathBuf::from(value));
            } else if let Some(value) = arg.to_str().and_then(|arg| arg.strip_prefix("--target=")) {
                target = Some(PathBuf::from(value));
            } else if let Some(value) = arg.to_str().and_then(|arg| arg.strip_prefix("--modified="))
            {
                modified = Some(PathBuf::from(value));
            } else {
                return Err(format!(
                    "unknown patch argument `{}`",
                    arg.to_string_lossy()
                ));
            }
            index += 1;
        }

        let profile = profile.ok_or_else(|| "--profile is required".to_string())?;
        let game_dir = game_dir.ok_or_else(|| "--game-dir is required".to_string())?;
        let target = target.ok_or_else(|| "--target is required".to_string())?;
        if needs_modified && modified.is_none() {
            return Err("--modified is required".to_string());
        }
        validate_relative_target(&target)?;

        Ok(Self {
            profile: sanitize_profile_name(&profile),
            game_dir,
            target,
            modified,
            dry_run,
        })
    }
}

fn patch_apply(args: &[OsString]) -> Result<(), String> {
    let args = PatchArgs::parse(args, true)?;
    let modified = args.modified.as_ref().expect("checked by parser");
    if !modified.is_file() {
        return Err(format!(
            "modified file does not exist: {}",
            modified.display()
        ));
    }

    let root = boost_root()?;
    let record = PatchRecord::new(&root, &args.profile, &args.game_dir, &args.target)?;
    let target_path = args.game_dir.join(&args.target);
    let had_original = target_path.exists();

    if target_path
        .symlink_metadata()
        .is_ok_and(|m| m.file_type().is_symlink())
    {
        return Err(format!(
            "refusing to patch symlink target: {}",
            target_path.display()
        ));
    }
    if record.manifest.is_file() {
        return Err(format!(
            "patch already active for `{}`; restore it before applying again",
            args.target.display()
        ));
    }

    println!("profile: {}", args.profile);
    println!("game dir: {}", args.game_dir.display());
    println!("target: {}", target_path.display());
    println!("modified: {}", modified.display());
    println!("backup: {}", record.backup.display());
    if args.dry_run {
        println!("dry run: no files changed");
        return Ok(());
    }

    fs::create_dir_all(record.backup.parent().expect("backup has parent"))
        .map_err(|err| format!("failed to create patch backup dir: {err}"))?;
    if let Some(parent) = target_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create target parent {}: {err}", parent.display()))?;
    }

    if had_original {
        fs::copy(&target_path, &record.backup).map_err(|err| {
            format!(
                "failed to back up {} to {}: {err}",
                target_path.display(),
                record.backup.display()
            )
        })?;
    }

    fs::copy(modified, &target_path).map_err(|err| {
        format!(
            "failed to copy modified file into {}: {err}",
            target_path.display()
        )
    })?;
    record
        .write_manifest(had_original, modified)
        .map_err(|err| format!("failed to write patch manifest: {err}"))?;

    println!(
        "patched `{}`; rollback manifest written",
        args.target.display()
    );
    Ok(())
}

fn patch_restore(args: &[OsString]) -> Result<(), String> {
    let args = PatchArgs::parse(args, false)?;
    let root = boost_root()?;
    let record = PatchRecord::new(&root, &args.profile, &args.game_dir, &args.target)?;
    if !record.manifest.is_file() {
        return Err(format!(
            "no active patch manifest for `{}`",
            args.target.display()
        ));
    }

    let manifest = fs::read_to_string(&record.manifest)
        .map_err(|err| format!("failed to read {}: {err}", record.manifest.display()))?;
    let had_original = manifest_bool(&manifest, "had_original").unwrap_or(false);
    let target_path = args.game_dir.join(&args.target);

    println!("profile: {}", args.profile);
    println!("target: {}", target_path.display());
    println!("backup: {}", record.backup.display());
    if args.dry_run {
        println!("dry run: no files changed");
        return Ok(());
    }

    if had_original {
        if !record.backup.is_file() {
            return Err(format!(
                "backup is missing, cannot restore safely: {}",
                record.backup.display()
            ));
        }
        fs::copy(&record.backup, &target_path)
            .map_err(|err| format!("failed to restore {}: {err}", target_path.display()))?;
    } else if target_path.exists() {
        fs::remove_file(&target_path).map_err(|err| {
            format!(
                "failed to remove patched file {}: {err}",
                target_path.display()
            )
        })?;
    }

    fs::remove_file(&record.manifest).map_err(|err| {
        format!(
            "failed to remove manifest {}: {err}",
            record.manifest.display()
        )
    })?;
    println!("restored `{}`", args.target.display());
    Ok(())
}

fn patch_status(args: &[OsString]) -> Result<(), String> {
    let mut profile_filter = None;
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        if arg == "--profile" {
            index += 1;
            profile_filter = args
                .get(index)
                .and_then(|value| value.to_str())
                .map(sanitize_profile_name);
        } else if let Some(value) = arg.to_str().and_then(|arg| arg.strip_prefix("--profile=")) {
            profile_filter = Some(sanitize_profile_name(value));
        } else {
            return Err(format!(
                "unknown patch status argument `{}`",
                arg.to_string_lossy()
            ));
        }
        index += 1;
    }

    let root = boost_root()?;
    let patch_root = root.join("patches");
    println!("Oxide Boost patches");
    println!("root: {}", patch_root.display());
    if !patch_root.is_dir() {
        println!("no active patches");
        return Ok(());
    }

    let mut found = false;
    for entry in
        fs::read_dir(&patch_root).map_err(|err| format!("failed to read patches: {err}"))?
    {
        let entry = entry.map_err(|err| format!("failed to read patch entry: {err}"))?;
        if !entry.path().is_dir() {
            continue;
        }
        let profile = entry.file_name().to_string_lossy().into_owned();
        if profile_filter
            .as_ref()
            .is_some_and(|filter| filter != &profile)
        {
            continue;
        }
        for manifest in find_manifests(&entry.path())? {
            found = true;
            println!("{}: {}", profile, manifest.display());
        }
    }
    if !found {
        println!("no active patches");
    }
    Ok(())
}

#[derive(Debug)]
struct PatchRecord {
    manifest: PathBuf,
    backup: PathBuf,
    profile: String,
    game_dir: PathBuf,
    target: PathBuf,
}

impl PatchRecord {
    fn new(root: &Path, profile: &str, game_dir: &Path, target: &Path) -> Result<Self, String> {
        let key = target_key(target);
        let patch_dir = root.join("patches").join(profile).join(&key);
        Ok(Self {
            manifest: patch_dir.join("manifest.json"),
            backup: patch_dir.join("original"),
            profile: profile.to_string(),
            game_dir: game_dir.to_path_buf(),
            target: target.to_path_buf(),
        })
    }

    fn write_manifest(&self, had_original: bool, modified: &Path) -> io::Result<()> {
        if let Some(parent) = self.manifest.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = File::create(&self.manifest)?;
        writeln!(file, "{{")?;
        writeln!(file, "  \"format\": \"oxide-boost-patch-v1\",")?;
        writeln!(file, "  \"profile\": \"{}\",", json_escape(&self.profile))?;
        writeln!(
            file,
            "  \"game_dir\": \"{}\",",
            json_escape(&self.game_dir.display().to_string())
        )?;
        writeln!(
            file,
            "  \"target\": \"{}\",",
            json_escape(&self.target.display().to_string())
        )?;
        writeln!(
            file,
            "  \"modified\": \"{}\",",
            json_escape(&modified.display().to_string())
        )?;
        writeln!(
            file,
            "  \"backup\": \"{}\",",
            json_escape(&self.backup.display().to_string())
        )?;
        writeln!(file, "  \"had_original\": {had_original},")?;
        writeln!(file, "  \"applied_ms\": {}", unix_ms())?;
        writeln!(file, "}}")?;
        Ok(())
    }
}

fn validate_relative_target(target: &Path) -> Result<(), String> {
    if target.is_absolute() {
        return Err("--target must be relative to --game-dir".to_string());
    }
    for component in target.components() {
        match component {
            Component::Normal(_) => {}
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err("--target may not escape --game-dir".to_string());
            }
        }
    }
    Ok(())
}

fn target_key(target: &Path) -> String {
    let display = target.display().to_string();
    format!(
        "{:016x}-{}",
        stable_hash(&display),
        sanitize_profile_name(&display)
    )
}

fn stable_hash(value: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in value.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn manifest_bool(text: &str, key: &str) -> Option<bool> {
    let needle = format!("\"{key}\": ");
    let start = text.find(&needle)? + needle.len();
    let line = text[start..]
        .trim_start()
        .lines()
        .next()?
        .trim_end_matches(',');
    match line {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

fn find_manifests(root: &Path) -> Result<Vec<PathBuf>, String> {
    let mut found = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(path) = stack.pop() {
        for entry in fs::read_dir(&path)
            .map_err(|err| format!("failed to read {}: {err}", path.display()))?
        {
            let entry =
                entry.map_err(|err| format!("failed to read {} entry: {err}", path.display()))?;
            let entry_path = entry.path();
            if entry_path.is_dir() {
                stack.push(entry_path);
            } else if entry_path
                .file_name()
                .is_some_and(|name| name == "manifest.json")
            {
                found.push(entry_path);
            }
        }
    }
    found.sort();
    Ok(found)
}

#[derive(Debug, Clone)]
struct BoostPaths {
    mesa_cache: PathBuf,
    dxvk_cache: PathBuf,
    run_dir: PathBuf,
}

impl BoostPaths {
    fn new(root: &Path, profile: &str) -> Self {
        let profile_root = root.join("profiles").join(profile);
        Self {
            mesa_cache: profile_root.join("mesa-shader-cache"),
            dxvk_cache: profile_root.join("dxvk-state-cache"),
            run_dir: root.join("runs"),
        }
    }

    fn create(&self) -> io::Result<()> {
        fs::create_dir_all(&self.mesa_cache)?;
        fs::create_dir_all(&self.dxvk_cache)?;
        fs::create_dir_all(&self.run_dir)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct CacheSnapshot {
    mesa_bytes: u64,
    dxvk_bytes: u64,
}

impl CacheSnapshot {
    fn read(paths: &BoostPaths) -> io::Result<Self> {
        Ok(Self {
            mesa_bytes: dir_size(&paths.mesa_cache)?,
            dxvk_bytes: dir_size(&paths.dxvk_cache)?,
        })
    }

    fn total_bytes(self) -> u64 {
        self.mesa_bytes + self.dxvk_bytes
    }
}

#[derive(Debug)]
struct RunRecord {
    profile: String,
    command: Vec<String>,
    started_ms: u128,
    elapsed_ms: u64,
    status_code: Option<i32>,
    before: CacheSnapshot,
    after: CacheSnapshot,
}

impl RunRecord {
    fn write(&self, root: &Path) -> io::Result<()> {
        let run_dir = root.join("runs");
        fs::create_dir_all(&run_dir)?;
        let path = run_dir.join(format!("{}-{}.json", self.started_ms, self.profile));
        let mut file = File::create(path)?;
        writeln!(file, "{{")?;
        writeln!(file, "  \"format\": \"oxide-boost-run-v1\",")?;
        writeln!(file, "  \"profile\": \"{}\",", json_escape(&self.profile))?;
        writeln!(file, "  \"started_ms\": {},", self.started_ms)?;
        writeln!(file, "  \"elapsed_ms\": {},", self.elapsed_ms)?;
        match self.status_code {
            Some(code) => writeln!(file, "  \"status_code\": {code},")?,
            None => writeln!(file, "  \"status_code\": null,")?,
        }
        writeln!(file, "  \"command\": [")?;
        for (index, arg) in self.command.iter().enumerate() {
            let comma = if index + 1 == self.command.len() {
                ""
            } else {
                ","
            };
            writeln!(file, "    \"{}\"{}", json_escape(arg), comma)?;
        }
        writeln!(file, "  ],")?;
        writeln!(file, "  \"cache_before\": {{")?;
        writeln!(file, "    \"mesa_bytes\": {},", self.before.mesa_bytes)?;
        writeln!(file, "    \"dxvk_bytes\": {}", self.before.dxvk_bytes)?;
        writeln!(file, "  }},")?;
        writeln!(file, "  \"cache_after\": {{")?;
        writeln!(file, "    \"mesa_bytes\": {},", self.after.mesa_bytes)?;
        writeln!(file, "    \"dxvk_bytes\": {}", self.after.dxvk_bytes)?;
        writeln!(file, "  }}")?;
        writeln!(file, "}}")?;
        Ok(())
    }
}

fn boost_root() -> Result<PathBuf, String> {
    if let Some(path) = env::var_os("OXIDE_BOOST_HOME") {
        return Ok(PathBuf::from(path));
    }
    let home = env::var_os("HOME").ok_or_else(|| {
        "HOME is not set; set OXIDE_BOOST_HOME to choose a cache root".to_string()
    })?;
    Ok(PathBuf::from(home).join(".cache").join("oxide-boost"))
}

fn infer_profile_name(program: &OsString) -> String {
    Path::new(program)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("default")
        .to_string()
}

fn sanitize_profile_name(name: &str) -> String {
    let sanitized = name
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' || ch == '.' {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>();
    if sanitized.is_empty() {
        "default".to_string()
    } else {
        sanitized
    }
}

fn dir_size(path: &Path) -> io::Result<u64> {
    if !path.exists() {
        return Ok(0);
    }
    let mut total = 0u64;
    let mut stack = vec![path.to_path_buf()];
    while let Some(path) = stack.pop() {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let metadata = entry.metadata()?;
            if metadata.is_dir() {
                stack.push(entry.path());
            } else if metadata.is_file() {
                total = total.saturating_add(metadata.len());
            }
        }
    }
    Ok(total)
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} {}", UNITS[unit])
    } else {
        format!("{value:.2} {}", UNITS[unit])
    }
}

fn unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn json_escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

#[cfg(test)]
mod tests {
    use super::{
        format_bytes, infer_profile_name, manifest_bool, sanitize_profile_name, target_key,
        validate_relative_target,
    };
    use std::ffi::OsString;
    use std::path::Path;

    #[test]
    fn sanitizes_profile_names_for_cache_paths() {
        assert_eq!(sanitize_profile_name("Cyber Game"), "Cyber_Game");
        assert_eq!(sanitize_profile_name("../bad"), ".._bad");
        assert_eq!(sanitize_profile_name(""), "default");
    }

    #[test]
    fn infers_profile_from_program_name() {
        assert_eq!(
            infer_profile_name(&OsString::from("/usr/bin/vkcube")),
            "vkcube"
        );
    }

    #[test]
    fn formats_cache_sizes() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(2048), "2.00 KiB");
    }

    #[test]
    fn rejects_targets_that_escape_game_dir() {
        assert!(validate_relative_target(Path::new("data/shader.bin")).is_ok());
        assert!(validate_relative_target(Path::new("../shader.bin")).is_err());
        assert!(validate_relative_target(Path::new("/tmp/shader.bin")).is_err());
    }

    #[test]
    fn patch_keys_are_stable_and_manifest_bool_parses() {
        assert_eq!(
            target_key(Path::new("a/b.txt")),
            target_key(Path::new("a/b.txt"))
        );
        let manifest = r#"{
  "had_original": true,
  "applied_ms": 1
}"#;
        assert_eq!(manifest_bool(manifest, "had_original"), Some(true));
    }
}
