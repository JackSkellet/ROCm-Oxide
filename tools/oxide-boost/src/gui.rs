use font8x8::{BASIC_FONTS, UnicodeFonts};
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Scale, Window, WindowOptions};
use std::ffi::OsString;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::thread;

const WIDTH: usize = 980;
const HEIGHT: usize = 620;

#[derive(Debug, Clone, Copy)]
struct Rect {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
}

impl Rect {
    fn contains(self, x: usize, y: usize) -> bool {
        x >= self.x && y >= self.y && x < self.x + self.w && y < self.y + self.h
    }
}

#[derive(Debug)]
struct TextField {
    label: &'static str,
    value: String,
    rect: Rect,
}

#[derive(Debug, Clone, Copy)]
enum ButtonAction {
    Apply,
    Restore,
    Status,
    DryRun,
    Clear,
    BrowseGameDir,
    BrowseModified,
    AnalyzeGame,
    EdgeAnalyze,
    GhidraReport,
    GenerateCandidate,
}

#[derive(Debug)]
struct Button {
    label: &'static str,
    rect: Rect,
    action: ButtonAction,
}

pub fn run() -> Result<(), String> {
    let mut window = Window::new(
        "Oxide Boost Patch GUI",
        WIDTH,
        HEIGHT,
        WindowOptions {
            resize: false,
            scale: Scale::X1,
            ..WindowOptions::default()
        },
    )
    .map_err(|err| format!("failed to create GUI window: {err}"))?;
    window.set_target_fps(60);

    let mut frame = vec![0u32; WIDTH * HEIGHT];
    let mut fields = vec![
        TextField {
            label: "Profile",
            value: "my-game".to_string(),
            rect: Rect {
                x: 230,
                y: 128,
                w: 690,
                h: 34,
            },
        },
        TextField {
            label: "Game Directory",
            value: String::new(),
            rect: Rect {
                x: 230,
                y: 180,
                w: 540,
                h: 34,
            },
        },
        TextField {
            label: "Target In Game",
            value: String::new(),
            rect: Rect {
                x: 230,
                y: 232,
                w: 690,
                h: 34,
            },
        },
        TextField {
            label: "Modified File",
            value: String::new(),
            rect: Rect {
                x: 230,
                y: 284,
                w: 540,
                h: 34,
            },
        },
    ];
    let buttons = vec![
        Button {
            label: "Apply Patch",
            rect: Rect {
                x: 230,
                y: 348,
                w: 150,
                h: 42,
            },
            action: ButtonAction::Apply,
        },
        Button {
            label: "Dry Run",
            rect: Rect {
                x: 396,
                y: 348,
                w: 126,
                h: 42,
            },
            action: ButtonAction::DryRun,
        },
        Button {
            label: "Restore",
            rect: Rect {
                x: 538,
                y: 348,
                w: 126,
                h: 42,
            },
            action: ButtonAction::Restore,
        },
        Button {
            label: "Status",
            rect: Rect {
                x: 680,
                y: 348,
                w: 116,
                h: 42,
            },
            action: ButtonAction::Status,
        },
        Button {
            label: "Clear Log",
            rect: Rect {
                x: 812,
                y: 348,
                w: 108,
                h: 42,
            },
            action: ButtonAction::Clear,
        },
        Button {
            label: "Game Dir...",
            rect: Rect {
                x: 790,
                y: 176,
                w: 130,
                h: 42,
            },
            action: ButtonAction::BrowseGameDir,
        },
        Button {
            label: "Mod File...",
            rect: Rect {
                x: 790,
                y: 280,
                w: 130,
                h: 42,
            },
            action: ButtonAction::BrowseModified,
        },
        Button {
            label: "Analyze Game",
            rect: Rect {
                x: 230,
                y: 404,
                w: 160,
                h: 42,
            },
            action: ButtonAction::AnalyzeGame,
        },
        Button {
            label: "Edge Analyze",
            rect: Rect {
                x: 404,
                y: 404,
                w: 150,
                h: 42,
            },
            action: ButtonAction::EdgeAnalyze,
        },
        Button {
            label: "Ghidra Report",
            rect: Rect {
                x: 570,
                y: 404,
                w: 160,
                h: 42,
            },
            action: ButtonAction::GhidraReport,
        },
        Button {
            label: "Generate File",
            rect: Rect {
                x: 746,
                y: 404,
                w: 174,
                h: 42,
            },
            action: ButtonAction::GenerateCandidate,
        },
    ];
    let mut focus = 0usize;
    let mut mouse_was_down = false;
    let mut status = "Fill the paths, then apply a reversible file patch.".to_string();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let mouse_down = window.get_mouse_down(MouseButton::Left);
        if mouse_down
            && !mouse_was_down
            && let Some((mx, my)) = window.get_mouse_pos(MouseMode::Discard)
        {
            let mx = mx as usize;
            let my = my as usize;
            for (index, field) in fields.iter().enumerate() {
                if field.rect.contains(mx, my) {
                    focus = index;
                }
            }
            for button in &buttons {
                if button.rect.contains(mx, my) {
                    status = handle_button(button.action, &mut fields, &mut focus);
                }
            }
        }
        mouse_was_down = mouse_down;

        let shift = window.is_key_down(Key::LeftShift) || window.is_key_down(Key::RightShift);
        for key in window.get_keys_pressed(KeyRepeat::Yes) {
            let ctrl = window.is_key_down(Key::LeftCtrl) || window.is_key_down(Key::RightCtrl);
            match key {
                Key::Tab => focus = (focus + 1) % fields.len(),
                Key::A if ctrl => {
                    fields[focus].value.clear();
                }
                Key::V if ctrl => match read_clipboard() {
                    Ok(text) => fields[focus].value.push_str(text.trim()),
                    Err(err) => status = format!("Paste failed: {err}"),
                },
                Key::Backspace => {
                    fields[focus].value.pop();
                }
                Key::Delete => fields[focus].value.clear(),
                Key::Enter => status = handle_button(ButtonAction::Apply, &mut fields, &mut focus),
                _ => {
                    if !ctrl
                        && let Some(ch) = key_char(key, shift)
                        && fields[focus].value.len() < 240
                    {
                        fields[focus].value.push(ch);
                    }
                }
            }
        }

        draw(&mut frame, &fields, &buttons, focus, &status);
        window
            .update_with_buffer(&frame, WIDTH, HEIGHT)
            .map_err(|err| format!("failed to update GUI window: {err}"))?;
    }

    Ok(())
}

pub fn analyze_game_dir_for_cli(game_dir: PathBuf) -> Result<String, String> {
    let analysis = analyze_game_dir(&game_dir)?;
    let plan = analysis
        .plan
        .as_ref()
        .map(|plan| format!("\n{}", plan.as_report()))
        .unwrap_or_default();
    Ok(format!(
        "engine analysis\nprofile: {}\ntarget: {}\nsummary: {}{}",
        analysis.profile,
        analysis
            .preferred_target
            .as_deref()
            .unwrap_or("<manual modified file required>"),
        analysis.summary,
        plan,
    ))
}

pub fn deep_analyze_game_dir_for_cli(
    game_dir: PathBuf,
    run_ghidra: bool,
) -> Result<String, String> {
    let mut out = analyze_game_dir_for_cli(game_dir.clone())?;
    out.push('\n');
    out.push_str("binary analysis\n");
    match ghidra_headless_path() {
        Some(path) => out.push_str(&format!("ghidra: available at {}\n", path.display())),
        None => out.push_str("ghidra: not found\n"),
    }

    for binary in binary_analysis_targets(&game_dir) {
        let relative = binary
            .strip_prefix(&game_dir)
            .unwrap_or(&binary)
            .display()
            .to_string();
        let clues = binary_clues(&binary);
        out.push_str(&format!("- {relative}: {}\n", clues.join(", ")));
    }
    if run_ghidra {
        match binary_analysis_targets(&game_dir).first() {
            Some(binary) => {
                let profile = analyze_game_dir_edge(&game_dir)?.profile;
                let report = run_ghidra_summary(binary, &profile)?;
                out.push_str(&format!("ghidra report: {}\n", report.display()));
            }
            None => out.push_str("ghidra report: no binary target found\n"),
        }
    }
    Ok(out)
}

pub fn analyze_game_dir_edge_for_cli(game_dir: PathBuf) -> Result<String, String> {
    let analysis = analyze_game_dir_edge(&game_dir)?;
    Ok(format!(
        "edge engine analysis\nprofile: {}\ntarget: {}\nsummary: {}",
        analysis.profile,
        analysis
            .preferred_target
            .as_deref()
            .unwrap_or("<manual modified file required>"),
        analysis.summary
    ))
}

pub fn ghidra_headless_path() -> Option<PathBuf> {
    find_in_path("analyzeHeadless").or_else(|| {
        [
            "/opt/ghidra/support/analyzeHeadless",
            "/usr/bin/analyzeHeadless",
        ]
        .iter()
        .map(PathBuf::from)
        .find(|path| path.is_file())
    })
}

fn handle_button(action: ButtonAction, fields: &mut [TextField], focus: &mut usize) -> String {
    match action {
        ButtonAction::Apply => apply_from_fields(fields, false),
        ButtonAction::DryRun => apply_from_fields(fields, true),
        ButtonAction::Restore => restore_from_fields(fields),
        ButtonAction::Status => status_from_fields(fields),
        ButtonAction::Clear => "Log cleared.".to_string(),
        ButtonAction::BrowseGameDir => match choose_directory() {
            Ok(Some(path)) => {
                fields[1].value = path.display().to_string();
                *focus = 1;
                "Selected game directory.".to_string()
            }
            Ok(None) => "Directory selection cancelled.".to_string(),
            Err(err) => format!("Directory picker failed: {err}"),
        },
        ButtonAction::BrowseModified => match choose_file() {
            Ok(Some(path)) => {
                fields[3].value = path.display().to_string();
                *focus = 3;
                "Selected modified file.".to_string()
            }
            Ok(None) => "File selection cancelled.".to_string(),
            Err(err) => format!("File picker failed: {err}"),
        },
        ButtonAction::AnalyzeGame => analyze_from_fields(fields),
        ButtonAction::EdgeAnalyze => edge_analyze_from_fields(fields),
        ButtonAction::GhidraReport => ghidra_report_from_fields(fields),
        ButtonAction::GenerateCandidate => generate_candidate_from_fields(fields),
    }
}

fn apply_from_fields(fields: &[TextField], dry_run: bool) -> String {
    let mut args = patch_base_args(fields);
    args.push(OsString::from("--modified"));
    args.push(OsString::from(fields[3].value.trim()));
    if dry_run {
        args.push(OsString::from("--dry-run"));
    }
    match super::patch_apply(&args) {
        Ok(()) if dry_run => "Dry run passed. No files changed.".to_string(),
        Ok(()) => "Patch applied. Original is backed up for restore.".to_string(),
        Err(err) => format!("Apply failed: {err}"),
    }
}

fn analyze_from_fields(fields: &mut [TextField]) -> String {
    let game_dir = PathBuf::from(fields[1].value.trim());
    match analyze_game_dir(&game_dir) {
        Ok(analysis) => {
            fields[0].value = analysis.profile;
            if let Some(target) = analysis.preferred_target {
                fields[2].value = target;
            }
            analysis.summary
        }
        Err(err) => format!("Analyze failed: {err}"),
    }
}

fn edge_analyze_from_fields(fields: &mut [TextField]) -> String {
    let game_dir = PathBuf::from(fields[1].value.trim());
    match analyze_game_dir_edge(&game_dir) {
        Ok(analysis) => {
            fields[0].value = analysis.profile;
            if let Some(target) = analysis.preferred_target {
                fields[2].value = target;
            }
            analysis.summary
        }
        Err(err) => format!("Edge analyze failed: {err}"),
    }
}

fn ghidra_report_from_fields(fields: &mut [TextField]) -> String {
    let game_dir = PathBuf::from(fields[1].value.trim());
    let profile = super::sanitize_profile_name(fields[0].value.trim());
    let target = PathBuf::from(fields[2].value.trim());
    let explicit_binary = (!fields[2].value.trim().is_empty())
        .then(|| game_dir.join(&target))
        .filter(|path| path.is_file());
    let binary = explicit_binary.or_else(|| binary_analysis_targets(&game_dir).into_iter().next());
    match binary {
        Some(binary) => match run_ghidra_summary(&binary, &profile) {
            Ok(report) => format!("Ghidra report written: {}", report.display()),
            Err(err) => format!("Ghidra report failed: {err}"),
        },
        None => "Ghidra report failed: no binary target found. Use Edge Analyze first.".to_string(),
    }
}

fn generate_candidate_from_fields(fields: &mut [TextField]) -> String {
    let game_dir = PathBuf::from(fields[1].value.trim());
    let profile = super::sanitize_profile_name(fields[0].value.trim());
    match generate_unity_boot_candidate(&game_dir, &profile) {
        Ok((target, modified, plan)) => {
            fields[2].value = target;
            fields[3].value = modified.display().to_string();
            if plan.changed_count() > 0 {
                format!(
                    "Generated Unity boot.config: {} change(s), {} already optimal.",
                    plan.changed_count(),
                    plan.unchanged_count()
                )
            } else {
                "Generated boot.config candidate; detected settings were already optimal."
                    .to_string()
            }
        }
        Err(err) => format!("Generate failed: {err}"),
    }
}

fn restore_from_fields(fields: &[TextField]) -> String {
    let args = patch_base_args(fields);
    match super::patch_restore(&args) {
        Ok(()) => "Original file restored from backup.".to_string(),
        Err(err) => format!("Restore failed: {err}"),
    }
}

fn status_from_fields(fields: &[TextField]) -> String {
    let profile = super::sanitize_profile_name(fields[0].value.trim());
    match super::boost_root() {
        Ok(root) => {
            let patch_root = root.join("patches").join(&profile);
            if !patch_root.is_dir() {
                return format!("No active patches for profile `{profile}`.");
            }
            match super::find_manifests(&patch_root) {
                Ok(found) if found.is_empty() => {
                    format!("No active patches for profile `{profile}`.")
                }
                Ok(found) => format!("{} active patch manifest(s) for `{profile}`.", found.len()),
                Err(err) => format!("Status failed: {err}"),
            }
        }
        Err(err) => format!("Status failed: {err}"),
    }
}

fn patch_base_args(fields: &[TextField]) -> Vec<OsString> {
    vec![
        OsString::from("--profile"),
        OsString::from(fields[0].value.trim()),
        OsString::from("--game-dir"),
        OsString::from(fields[1].value.trim()),
        OsString::from("--target"),
        OsString::from(fields[2].value.trim()),
    ]
}

fn draw(frame: &mut [u32], fields: &[TextField], buttons: &[Button], focus: usize, status: &str) {
    frame.fill(0x0f141b);
    draw_rect(frame, 0, 0, WIDTH, 84, 0x17202a);
    draw_rect(frame, 0, 84, WIDTH, 2, 0x2fa7d6);
    draw_text(frame, 34, 22, "Oxide Boost", 0xffffff);
    draw_text(
        frame,
        34,
        48,
        "reversible game file patching for performance experiments",
        0x9fd7ff,
    );

    draw_rect(frame, 34, 116, 160, 280, 0x151b22);
    draw_rect_outline(frame, 34, 116, 160, 280, 0x304658);
    draw_text(frame, 50, 136, "Patch Flow", 0xffffff);
    draw_text(frame, 50, 168, "1 choose profile", 0xb8c7d6);
    draw_text(frame, 50, 194, "2 select game dir", 0xb8c7d6);
    draw_text(frame, 50, 220, "3 target relative", 0xb8c7d6);
    draw_text(frame, 50, 246, "4 modified file", 0xb8c7d6);
    draw_text(frame, 50, 286, "Ctrl+V paste", 0x7d8ca0);
    draw_text(frame, 50, 312, "Ctrl+A clear", 0x7d8ca0);
    draw_text(frame, 50, 346, "Esc exits", 0x7d8ca0);

    for (index, field) in fields.iter().enumerate() {
        draw_text(frame, 230, field.rect.y - 20, field.label, 0xdce8f2);
        let fill = if index == focus { 0x263647 } else { 0x18222d };
        draw_rect(
            frame,
            field.rect.x,
            field.rect.y,
            field.rect.w,
            field.rect.h,
            fill,
        );
        draw_rect_outline(
            frame,
            field.rect.x,
            field.rect.y,
            field.rect.w,
            field.rect.h,
            if index == focus { 0x5cc8ff } else { 0x4c6578 },
        );
        let shown = elide_left(&field.value, 82);
        draw_text(
            frame,
            field.rect.x + 10,
            field.rect.y + 10,
            &shown,
            0xf3f6fa,
        );
    }

    for button in buttons {
        draw_rect(
            frame,
            button.rect.x,
            button.rect.y,
            button.rect.w,
            button.rect.h,
            0x243545,
        );
        draw_rect_outline(
            frame,
            button.rect.x,
            button.rect.y,
            button.rect.w,
            button.rect.h,
            0x7896aa,
        );
        draw_text(
            frame,
            button.rect.x + 12,
            button.rect.y + 14,
            button.label,
            0xffffff,
        );
    }

    draw_rect(frame, 34, 468, WIDTH - 68, 108, 0x141a22);
    draw_rect_outline(frame, 34, 468, WIDTH - 68, 108, 0x455b6f);
    draw_text(frame, 52, 488, "Status", 0xffffff);
    for (line, text) in wrap_text(status, 104).iter().take(4).enumerate() {
        draw_text(frame, 52, 520 + line * 18, text, 0xffd08a);
    }
}

#[derive(Debug)]
struct GameAnalysis {
    profile: String,
    preferred_target: Option<String>,
    summary: String,
    plan: Option<OptimizationPlan>,
}

fn analyze_game_dir(game_dir: &PathBuf) -> Result<GameAnalysis, String> {
    if !game_dir.is_dir() {
        return Err(format!("not a directory: {}", game_dir.display()));
    }

    let unity_data = find_unity_data_dir(game_dir);
    if let Some(data_dir) = unity_data {
        let app = read_unity_app_info(&data_dir);
        let profile = app.product.unwrap_or_else(|| {
            game_dir
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("unity-game")
                .to_string()
        });
        let rel_boot = data_dir
            .strip_prefix(game_dir)
            .ok()
            .map(|path| path.join("boot.config").display().to_string());
        let il2cpp =
            game_dir.join("GameAssembly.so").is_file() || data_dir.join("il2cpp_data").is_dir();
        let plugins = data_dir.join("Plugins").is_dir();
        let boot_summary = summarize_boot_config(&data_dir.join("boot.config"));
        let plan = build_unity_boot_plan(game_dir)?;
        let plan_summary = format!(
            " Plan: {} change(s), {} already optimal.",
            plan.changed_count(),
            plan.unchanged_count()
        );
        let vendor = app
            .company
            .as_deref()
            .filter(|value| !value.is_empty())
            .unwrap_or("unknown vendor");
        return Ok(GameAnalysis {
            profile: super::sanitize_profile_name(&profile),
            preferred_target: rel_boot,
            summary: format!(
                "Detected Unity{} build by {vendor}. {} Plugins dir: {}. Preferred safe target: boot.config.{}",
                if il2cpp { " IL2CPP" } else { "" },
                boot_summary,
                if plugins { "yes" } else { "no" },
                plan_summary,
            ),
            plan: Some(plan),
        });
    }

    if game_dir.join("project.godot").is_file() || find_file_with_ext(game_dir, "pck", 2).is_some()
    {
        return Ok(GameAnalysis {
            profile: super::sanitize_profile_name(
                game_dir
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("godot-game"),
            ),
            preferred_target: find_file_with_ext(game_dir, "pck", 2),
            summary: "Detected likely Godot layout. Use replacement .pck/config patches; automatic binary edits are disabled.".to_string(),
            plan: None,
        });
    }

    if find_file_with_ext(game_dir, "pak", 3).is_some() {
        return Ok(GameAnalysis {
            profile: super::sanitize_profile_name(
                game_dir
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("unreal-game"),
            ),
            preferred_target: find_file_with_ext(game_dir, "ini", 4),
            summary: "Detected likely Unreal/package layout. Prefer Engine.ini/GameUserSettings.ini replacement patches.".to_string(),
            plan: None,
        });
    }

    Ok(GameAnalysis {
        profile: super::sanitize_profile_name(
            game_dir
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("game"),
        ),
        preferred_target: find_file_with_ext(game_dir, "cfg", 3)
            .or_else(|| find_file_with_ext(game_dir, "ini", 3)),
        summary:
            "Unknown engine. Found no safe automatic engine patch; use a supplied modified file."
                .to_string(),
        plan: None,
    })
}

fn analyze_game_dir_edge(game_dir: &PathBuf) -> Result<GameAnalysis, String> {
    let mut analysis = analyze_game_dir(game_dir)?;
    let binaries = binary_analysis_targets(game_dir);
    if let Some(first_binary) = binaries.first()
        && let Ok(relative) = first_binary.strip_prefix(game_dir)
    {
        analysis.preferred_target = Some(relative.display().to_string());
    }
    let ghidra = ghidra_headless_path()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "not found".to_string());
    let target_list = binaries
        .iter()
        .filter_map(|path| path.strip_prefix(game_dir).ok())
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(", ");
    analysis.summary = format!(
        "EDGE mode: binary/runtime targets enabled. Ghidra: {ghidra}. Candidate targets: {}. Provide a modified replacement file before applying.",
        if target_list.is_empty() {
            "<none found>".to_string()
        } else {
            target_list
        }
    );
    Ok(analysis)
}

#[derive(Debug, Clone)]
struct OptimizationPlan {
    target: String,
    generated_text: String,
    changes: Vec<PlanChange>,
    system: SystemProfile,
}

impl OptimizationPlan {
    fn changed_count(&self) -> usize {
        self.changes
            .iter()
            .filter(|change| change.current.as_deref() != Some(change.proposed.as_str()))
            .count()
    }

    fn unchanged_count(&self) -> usize {
        self.changes.len().saturating_sub(self.changed_count())
    }

    fn as_report(&self) -> String {
        let mut out = format!(
            "optimization plan\nsystem: {} physical cores, {} logical threads\ntarget: {}\n",
            self.system.physical_cores, self.system.logical_threads, self.target
        );
        for change in &self.changes {
            let state = if change.current.as_deref() == Some(change.proposed.as_str()) {
                "keep"
            } else if change.current.is_some() {
                "change"
            } else {
                "add"
            };
            out.push_str(&format!(
                "- {state}: {}={} ({}, confidence {})\n",
                change.key, change.proposed, change.reason, change.confidence
            ));
        }
        out
    }
}

#[derive(Debug, Clone)]
struct PlanChange {
    key: String,
    current: Option<String>,
    proposed: String,
    reason: String,
    confidence: &'static str,
}

#[derive(Debug, Clone)]
struct SystemProfile {
    logical_threads: usize,
    physical_cores: usize,
}

impl SystemProfile {
    fn detect() -> Self {
        let logical_threads = thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(4);
        let physical_cores =
            detect_physical_cores().unwrap_or_else(|| (logical_threads / 2).max(1));
        Self {
            logical_threads,
            physical_cores,
        }
    }
}

#[derive(Debug, Default)]
struct UnityAppInfo {
    company: Option<String>,
    product: Option<String>,
}

fn read_unity_app_info(data_dir: &PathBuf) -> UnityAppInfo {
    let Ok(text) = fs::read_to_string(data_dir.join("app.info")) else {
        return UnityAppInfo::default();
    };
    let mut lines = text.lines();
    UnityAppInfo {
        company: lines.next().map(str::to_string),
        product: lines.next().map(str::to_string),
    }
}

fn summarize_boot_config(path: &PathBuf) -> String {
    let Ok(text) = fs::read_to_string(path) else {
        return "No boot.config found.".to_string();
    };
    let gfx_jobs = has_boot_key_value(&text, "gfx-enable-gfx-jobs", "1");
    let native_jobs = has_boot_key_value(&text, "gfx-enable-native-gfx-jobs", "1");
    if gfx_jobs && native_jobs {
        "Unity graphics jobs already enabled.".to_string()
    } else {
        "Unity graphics jobs can be enabled in boot.config.".to_string()
    }
}

fn generate_unity_boot_candidate(
    game_dir: &PathBuf,
    profile: &str,
) -> Result<(String, PathBuf, OptimizationPlan), String> {
    let plan = build_unity_boot_plan(game_dir)?;
    let target = plan.target.clone();
    let root = super::boost_root()?;
    let out_dir = root.join("generated").join(profile);
    fs::create_dir_all(&out_dir)
        .map_err(|err| format!("failed to create {}: {err}", out_dir.display()))?;
    let out_path = out_dir.join("boot.config");
    fs::write(&out_path, &plan.generated_text)
        .map_err(|err| format!("failed to write {}: {err}", out_path.display()))?;
    Ok((target, out_path, plan))
}

fn build_unity_boot_plan(game_dir: &PathBuf) -> Result<OptimizationPlan, String> {
    let data_dir = find_unity_data_dir(game_dir)
        .ok_or_else(|| "no Unity *_Data directory with boot.config found".to_string())?;
    let source = data_dir.join("boot.config");
    let text = fs::read_to_string(&source)
        .map_err(|err| format!("failed to read {}: {err}", source.display()))?;
    let system = SystemProfile::detect();
    let target = data_dir
        .strip_prefix(game_dir)
        .map_err(|_| "Unity data directory is not inside game directory".to_string())?
        .join("boot.config")
        .display()
        .to_string();
    let desired = unity_boot_desired_settings(&text, &system);
    let changes = desired
        .iter()
        .map(|setting| PlanChange {
            key: setting.key.to_string(),
            current: boot_config_value(&text, setting.key),
            proposed: setting.value.clone(),
            reason: setting.reason.clone(),
            confidence: setting.confidence,
        })
        .collect::<Vec<_>>();
    let generated_text = ensure_boot_config_values_owned(&text, &desired);
    Ok(OptimizationPlan {
        target,
        generated_text,
        changes,
        system,
    })
}

#[derive(Debug, Clone)]
struct DesiredBootSetting {
    key: &'static str,
    value: String,
    reason: String,
    confidence: &'static str,
}

fn unity_boot_desired_settings(text: &str, system: &SystemProfile) -> Vec<DesiredBootSetting> {
    let physical = system.physical_cores.max(1);
    let worker_count = physical.saturating_sub(2).clamp(2, 16);
    let background_workers = (physical / 4).clamp(1, 4);
    let gc_slice = boot_config_value(text, "gc-max-time-slice")
        .and_then(|value| value.parse::<u32>().ok())
        .map(|value| value.clamp(1, 5))
        .unwrap_or(3);

    vec![
        DesiredBootSetting {
            key: "gfx-enable-gfx-jobs",
            value: "1".to_string(),
            reason: "Unity render work can be scheduled through the job system".to_string(),
            confidence: "high",
        },
        DesiredBootSetting {
            key: "gfx-enable-native-gfx-jobs",
            value: "1".to_string(),
            reason:
                "native render-thread jobs are beneficial for IL2CPP/Linux builds when supported"
                    .to_string(),
            confidence: "high",
        },
        DesiredBootSetting {
            key: "gfx-threading-mode",
            value: "4".to_string(),
            reason: "current game already uses Unity's threaded graphics mode; preserve it"
                .to_string(),
            confidence: "medium",
        },
        DesiredBootSetting {
            key: "job-worker-count",
            value: worker_count.to_string(),
            reason: format!(
                "computed from {physical} physical cores, reserving CPU time for render/audio/OS threads"
            ),
            confidence: "medium",
        },
        DesiredBootSetting {
            key: "background-job-worker-count",
            value: background_workers.to_string(),
            reason: "keeps asset/background jobs from competing with foreground frame work"
                .to_string(),
            confidence: "medium",
        },
        DesiredBootSetting {
            key: "gc-max-time-slice",
            value: gc_slice.to_string(),
            reason: "keeps Unity incremental GC in a conservative frame-time budget".to_string(),
            confidence: "medium",
        },
        DesiredBootSetting {
            key: "wait-for-native-debugger",
            value: "0".to_string(),
            reason: "ensures release startup path never waits for debugger attachment".to_string(),
            confidence: "high",
        },
        DesiredBootSetting {
            key: "hdr-display-enabled",
            value: "0".to_string(),
            reason: "avoids HDR output path unless the game explicitly ships with it enabled"
                .to_string(),
            confidence: "low",
        },
    ]
}

fn ensure_boot_config_values(text: &str, values: &[(&str, &str)]) -> String {
    let mut lines = text.lines().map(str::to_string).collect::<Vec<_>>();
    for (key, value) in values {
        let prefix = format!("{key}=");
        if let Some(line) = lines.iter_mut().find(|line| line.starts_with(&prefix)) {
            *line = format!("{key}={value}");
        } else {
            lines.push(format!("{key}={value}"));
        }
    }
    let mut out = lines.join("\n");
    out.push('\n');
    out
}

fn ensure_boot_config_values_owned(text: &str, values: &[DesiredBootSetting]) -> String {
    let pairs = values
        .iter()
        .map(|setting| (setting.key, setting.value.as_str()))
        .collect::<Vec<_>>();
    ensure_boot_config_values(text, &pairs)
}

fn boot_config_value(text: &str, key: &str) -> Option<String> {
    let prefix = format!("{key}=");
    text.lines()
        .find_map(|line| line.trim().strip_prefix(&prefix).map(str::to_string))
}

fn has_boot_key_value(text: &str, key: &str, value: &str) -> bool {
    let expected = format!("{key}={value}");
    text.lines().any(|line| line.trim() == expected)
}

fn detect_physical_cores() -> Option<usize> {
    let text = fs::read_to_string("/proc/cpuinfo").ok()?;
    let mut physical_ids = Vec::<(String, String)>::new();
    let mut current_physical = None::<String>;
    let mut current_core = None::<String>;
    for line in text.lines().chain([""]) {
        if line.trim().is_empty() {
            if let (Some(physical), Some(core)) = (current_physical.take(), current_core.take())
                && !physical_ids.contains(&(physical.clone(), core.clone()))
            {
                physical_ids.push((physical, core));
            }
            current_physical = None;
            current_core = None;
            continue;
        }
        if let Some(value) = line.strip_prefix("physical id") {
            current_physical = value
                .split(':')
                .nth(1)
                .map(|value| value.trim().to_string());
        } else if let Some(value) = line.strip_prefix("core id") {
            current_core = value
                .split(':')
                .nth(1)
                .map(|value| value.trim().to_string());
        }
    }
    if physical_ids.is_empty() {
        None
    } else {
        Some(physical_ids.len())
    }
}

fn find_unity_data_dir(game_dir: &PathBuf) -> Option<PathBuf> {
    fs::read_dir(game_dir)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .find(|path| {
            path.is_dir()
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.ends_with("_Data"))
                && path.join("boot.config").is_file()
                && path.join("globalgamemanagers").is_file()
        })
}

fn find_file_with_ext(game_dir: &PathBuf, ext: &str, max_depth: usize) -> Option<String> {
    let mut stack = vec![(game_dir.clone(), 0usize)];
    while let Some((dir, depth)) = stack.pop() {
        if depth > max_depth {
            continue;
        }
        for entry in fs::read_dir(&dir).ok()?.filter_map(Result::ok) {
            let path = entry.path();
            if path.is_dir() {
                stack.push((path, depth + 1));
            } else if path
                .extension()
                .and_then(|value| value.to_str())
                .is_some_and(|value| value.eq_ignore_ascii_case(ext))
                && let Ok(relative) = path.strip_prefix(game_dir)
            {
                return Some(relative.display().to_string());
            }
        }
    }
    None
}

fn read_clipboard() -> Result<String, String> {
    let mut clipboard =
        arboard::Clipboard::new().map_err(|err| format!("clipboard unavailable: {err}"))?;
    clipboard
        .get_text()
        .map_err(|err| format!("clipboard text unavailable: {err}"))
}

fn choose_directory() -> Result<Option<PathBuf>, String> {
    choose_path(&[
        "--file-selection",
        "--directory",
        "--title=Choose game directory",
    ])
}

fn choose_file() -> Result<Option<PathBuf>, String> {
    choose_path(&["--file-selection", "--title=Choose modified file"])
}

fn choose_path(args: &[&str]) -> Result<Option<PathBuf>, String> {
    let output = Command::new("zenity")
        .args(args)
        .output()
        .map_err(|err| format!("failed to launch zenity: {err}"))?;
    if output.status.success() {
        let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if text.is_empty() {
            Ok(None)
        } else {
            Ok(Some(PathBuf::from(text)))
        }
    } else {
        Ok(None)
    }
}

fn binary_analysis_targets(game_dir: &PathBuf) -> Vec<PathBuf> {
    [
        game_dir.join("GameAssembly.so"),
        game_dir.join("UnityPlayer.so"),
        game_dir.join("GameAssembly.dll"),
        game_dir.join("UnityPlayer.dll"),
    ]
    .into_iter()
    .filter(|path| path.is_file())
    .collect()
}

fn binary_clues(binary: &PathBuf) -> Vec<String> {
    let mut clues = Vec::new();
    let strings = Command::new("strings")
        .arg("-n")
        .arg("6")
        .arg(binary)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).into_owned())
        .unwrap_or_default();
    let symbols = Command::new("readelf")
        .arg("-Ws")
        .arg(binary)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).into_owned())
        .unwrap_or_default();
    let haystack = format!("{strings}\n{symbols}");

    if haystack.contains("il2cpp_gc_set_max_time_slice_ns") {
        clues.push("IL2CPP GC time-slice control exported".to_string());
    }
    if haystack.contains("Unity.Jobs") || haystack.contains("JobsUtility") {
        clues.push("Unity job-system symbols present".to_string());
    }
    if haystack.contains("pthread_create") {
        clues.push("native thread creation/imports present".to_string());
    }
    if haystack.to_ascii_lowercase().contains("vulkan") {
        clues.push("Vulkan-related strings present".to_string());
    }
    if clues.is_empty() {
        clues.push("no performance-related strings found by fast scan".to_string());
    }
    clues
}

fn run_ghidra_summary(binary: &PathBuf, profile: &str) -> Result<PathBuf, String> {
    if !binary.is_file() {
        return Err(format!(
            "binary target does not exist: {}",
            binary.display()
        ));
    }
    let ghidra = ghidra_headless_path()
        .ok_or_else(|| "Ghidra analyzeHeadless was not found in PATH or /opt/ghidra".to_string())?;
    let timeout = find_in_path("timeout");
    let root = super::boost_root()?;
    let safe_profile = super::sanitize_profile_name(profile);
    let binary_name = binary
        .file_name()
        .and_then(|name| name.to_str())
        .map(super::sanitize_profile_name)
        .unwrap_or_else(|| "binary".to_string());
    let ghidra_root = root.join("ghidra").join(&safe_profile);
    let project_dir = std::env::temp_dir()
        .join("oxide-boost-ghidra")
        .join(&safe_profile)
        .join("project");
    let report_dir = ghidra_root.join("reports");
    let script_dir = ghidra_root.join("scripts");
    fs::create_dir_all(&project_dir)
        .map_err(|err| format!("failed to create {}: {err}", project_dir.display()))?;
    fs::create_dir_all(&report_dir)
        .map_err(|err| format!("failed to create {}: {err}", report_dir.display()))?;
    fs::create_dir_all(&script_dir)
        .map_err(|err| format!("failed to create {}: {err}", script_dir.display()))?;

    let script_path = script_dir.join("OxideBoostGhidraSummary.java");
    fs::write(&script_path, GHIDRA_SUMMARY_SCRIPT)
        .map_err(|err| format!("failed to write {}: {err}", script_path.display()))?;
    let report_path = report_dir.join(format!("{binary_name}.summary.txt"));
    let log_path = report_dir.join(format!("{binary_name}.ghidra.log"));
    let script_log_path = report_dir.join(format!("{binary_name}.script.log"));
    let project_name = format!("oxide_boost_{safe_profile}_{binary_name}");

    let mut args = Vec::<OsString>::new();
    args.push(project_dir.clone().into_os_string());
    args.push(OsString::from(project_name));
    args.push(OsString::from("-import"));
    args.push(binary.clone().into_os_string());
    args.push(OsString::from("-overwrite"));
    args.push(OsString::from("-deleteProject"));
    args.push(OsString::from("-analysisTimeoutPerFile"));
    args.push(OsString::from("75"));
    args.push(OsString::from("-max-cpu"));
    args.push(OsString::from("4"));
    args.push(OsString::from("-scriptPath"));
    args.push(script_dir.clone().into_os_string());
    args.push(OsString::from("-postScript"));
    args.push(OsString::from("OxideBoostGhidraSummary.java"));
    args.push(report_path.clone().into_os_string());
    args.push(OsString::from("-log"));
    args.push(log_path.clone().into_os_string());
    args.push(OsString::from("-scriptlog"));
    args.push(script_log_path.clone().into_os_string());

    let output = if let Some(timeout) = timeout {
        let mut command = Command::new(timeout);
        command
            .arg("--foreground")
            .arg("120")
            .arg(&ghidra)
            .args(&args);
        command.output()
    } else {
        Command::new(&ghidra).args(&args).output()
    }
    .map_err(|err| format!("failed to launch Ghidra: {err}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let log_tail = tail_file(&log_path, 20).unwrap_or_default();
        return Err(format!(
            "Ghidra exited with status {}. stdout: {} stderr: {} log tail: {}",
            output.status,
            compact_one_line(&stdout),
            compact_one_line(&stderr),
            compact_one_line(&log_tail)
        ));
    }
    if !report_path.is_file() {
        return Err(format!(
            "Ghidra finished but did not write report: {}",
            report_path.display()
        ));
    }
    Ok(report_path)
}

fn tail_file(path: &PathBuf, lines: usize) -> Result<String, String> {
    let text = fs::read_to_string(path)
        .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    let mut tail = text.lines().rev().take(lines).collect::<Vec<_>>();
    tail.reverse();
    Ok(tail.join("\n"))
}

fn compact_one_line(text: &str) -> String {
    let compact = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if compact.len() > 500 {
        format!("{}...", &compact[..500])
    } else {
        compact
    }
}

const GHIDRA_SUMMARY_SCRIPT: &str = r#"// Oxide Boost generated helper.
// Writes a small performance-oriented symbol/function summary for a binary.
import ghidra.app.script.GhidraScript;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionIterator;
import ghidra.program.model.symbol.Symbol;
import ghidra.program.model.symbol.SymbolIterator;
import java.io.File;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.List;

public class OxideBoostGhidraSummary extends GhidraScript {
    @Override
    protected void run() throws Exception {
        String[] args = getScriptArgs();
        File outFile = new File(args.length > 0 ? args[0] : "oxide-boost-ghidra-summary.txt");
        File parent = outFile.getParentFile();
        if (parent != null) {
            parent.mkdirs();
        }

        List<String> needles = Arrays.asList(
            "il2cpp_gc", "gc_set", "gc_collect", "jobhandle", "jobsutility",
            "schedulebatchedjobs", "worker", "thread", "pthread", "vulkan",
            "present", "shader", "render", "ray", "trace", "physics"
        );

        try (PrintWriter out = new PrintWriter(outFile)) {
            out.println("program: " + currentProgram.getName());
            out.println("language: " + currentProgram.getLanguageID());
            out.println("compiler: " + currentProgram.getCompilerSpec().getCompilerSpecID());
            out.println("image_base: " + currentProgram.getImageBase());
            out.println("min_address: " + currentProgram.getMinAddress());
            out.println("max_address: " + currentProgram.getMaxAddress());
            out.println("function_count: " + currentProgram.getFunctionManager().getFunctionCount());
            out.println();
            out.println("symbol_hits:");
            int symbolHits = 0;
            SymbolIterator symbols = currentProgram.getSymbolTable().getAllSymbols(true);
            while (symbols.hasNext() && symbolHits < 250) {
                Symbol symbol = symbols.next();
                String name = symbol.getName(true);
                String lower = name.toLowerCase();
                for (String needle : needles) {
                    if (lower.contains(needle)) {
                        out.println(symbol.getAddress() + " " + name);
                        symbolHits++;
                        break;
                    }
                }
            }
            out.println();
            out.println("function_hits:");
            int functionHits = 0;
            FunctionIterator functions = currentProgram.getFunctionManager().getFunctions(true);
            while (functions.hasNext() && functionHits < 250) {
                Function function = functions.next();
                String name = function.getName(true);
                String lower = name.toLowerCase();
                for (String needle : needles) {
                    if (lower.contains(needle)) {
                        out.println(function.getEntryPoint() + " " + name);
                        functionHits++;
                        break;
                    }
                }
            }
        }
    }
}
"#;

fn find_in_path(program: &str) -> Option<PathBuf> {
    std::env::var_os("PATH").and_then(|path| {
        std::env::split_paths(&path)
            .map(|dir| dir.join(program))
            .find(|path| path.is_file())
    })
}

#[cfg(test)]
mod tests {
    use super::{ensure_boot_config_values, has_boot_key_value, wrap_text};

    #[test]
    fn boot_config_values_are_inserted_or_replaced() {
        let input = "gfx-enable-gfx-jobs=0\nbuild-guid=abc\n";
        let output = ensure_boot_config_values(
            input,
            &[
                ("gfx-enable-gfx-jobs", "1"),
                ("gfx-enable-native-gfx-jobs", "1"),
            ],
        );
        assert!(has_boot_key_value(&output, "gfx-enable-gfx-jobs", "1"));
        assert!(has_boot_key_value(
            &output,
            "gfx-enable-native-gfx-jobs",
            "1"
        ));
        assert!(output.contains("build-guid=abc"));
    }

    #[test]
    fn text_wrapping_keeps_short_lines() {
        let lines = wrap_text("one two three four", 7);
        assert_eq!(lines, vec!["one two", "three", "four"]);
    }
}

fn key_char(key: Key, shift: bool) -> Option<char> {
    match key {
        Key::A => Some(if shift { 'A' } else { 'a' }),
        Key::B => Some(if shift { 'B' } else { 'b' }),
        Key::C => Some(if shift { 'C' } else { 'c' }),
        Key::D => Some(if shift { 'D' } else { 'd' }),
        Key::E => Some(if shift { 'E' } else { 'e' }),
        Key::F => Some(if shift { 'F' } else { 'f' }),
        Key::G => Some(if shift { 'G' } else { 'g' }),
        Key::H => Some(if shift { 'H' } else { 'h' }),
        Key::I => Some(if shift { 'I' } else { 'i' }),
        Key::J => Some(if shift { 'J' } else { 'j' }),
        Key::K => Some(if shift { 'K' } else { 'k' }),
        Key::L => Some(if shift { 'L' } else { 'l' }),
        Key::M => Some(if shift { 'M' } else { 'm' }),
        Key::N => Some(if shift { 'N' } else { 'n' }),
        Key::O => Some(if shift { 'O' } else { 'o' }),
        Key::P => Some(if shift { 'P' } else { 'p' }),
        Key::Q => Some(if shift { 'Q' } else { 'q' }),
        Key::R => Some(if shift { 'R' } else { 'r' }),
        Key::S => Some(if shift { 'S' } else { 's' }),
        Key::T => Some(if shift { 'T' } else { 't' }),
        Key::U => Some(if shift { 'U' } else { 'u' }),
        Key::V => Some(if shift { 'V' } else { 'v' }),
        Key::W => Some(if shift { 'W' } else { 'w' }),
        Key::X => Some(if shift { 'X' } else { 'x' }),
        Key::Y => Some(if shift { 'Y' } else { 'y' }),
        Key::Z => Some(if shift { 'Z' } else { 'z' }),
        Key::Key0 | Key::NumPad0 => Some(if shift { ')' } else { '0' }),
        Key::Key1 | Key::NumPad1 => Some(if shift { '!' } else { '1' }),
        Key::Key2 | Key::NumPad2 => Some(if shift { '@' } else { '2' }),
        Key::Key3 | Key::NumPad3 => Some(if shift { '#' } else { '3' }),
        Key::Key4 | Key::NumPad4 => Some(if shift { '$' } else { '4' }),
        Key::Key5 | Key::NumPad5 => Some(if shift { '%' } else { '5' }),
        Key::Key6 | Key::NumPad6 => Some(if shift { '^' } else { '6' }),
        Key::Key7 | Key::NumPad7 => Some(if shift { '&' } else { '7' }),
        Key::Key8 | Key::NumPad8 => Some(if shift { '*' } else { '8' }),
        Key::Key9 | Key::NumPad9 => Some(if shift { '(' } else { '9' }),
        Key::Space => Some(' '),
        Key::Slash | Key::NumPadSlash => Some(if shift { '?' } else { '/' }),
        Key::Backslash => Some(if shift { '|' } else { '\\' }),
        Key::Minus | Key::NumPadMinus => Some(if shift { '_' } else { '-' }),
        Key::Period | Key::NumPadDot => Some('.'),
        Key::Comma => Some(','),
        Key::Equal => Some(if shift { '+' } else { '=' }),
        Key::Semicolon => Some(if shift { ':' } else { ';' }),
        Key::Apostrophe => Some(if shift { '"' } else { '\'' }),
        Key::LeftBracket => Some(if shift { '{' } else { '[' }),
        Key::RightBracket => Some(if shift { '}' } else { ']' }),
        Key::Backquote => Some(if shift { '~' } else { '`' }),
        _ => None,
    }
}

fn wrap_text(text: &str, max_chars: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        if !current.is_empty() && current.len() + word.len() + 1 > max_chars {
            lines.push(current);
            current = String::new();
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(word);
    }
    if !current.is_empty() {
        lines.push(current);
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

fn elide_left(value: &str, max_chars: usize) -> String {
    let count = value.chars().count();
    if count <= max_chars {
        value.to_string()
    } else {
        format!(
            "...{}",
            value
                .chars()
                .skip(count - max_chars + 3)
                .collect::<String>()
        )
    }
}

fn draw_rect(frame: &mut [u32], x: usize, y: usize, w: usize, h: usize, color: u32) {
    let max_y = (y + h).min(HEIGHT);
    let max_x = (x + w).min(WIDTH);
    for py in y.min(HEIGHT)..max_y {
        let row = py * WIDTH;
        for px in x.min(WIDTH)..max_x {
            frame[row + px] = color;
        }
    }
}

fn draw_rect_outline(frame: &mut [u32], x: usize, y: usize, w: usize, h: usize, color: u32) {
    draw_rect(frame, x, y, w, 1, color);
    draw_rect(frame, x, y + h.saturating_sub(1), w, 1, color);
    draw_rect(frame, x, y, 1, h, color);
    draw_rect(frame, x + w.saturating_sub(1), y, 1, h, color);
}

fn draw_text(frame: &mut [u32], x: usize, y: usize, text: &str, color: u32) {
    for (char_index, ch) in text.chars().enumerate() {
        if let Some(glyph) = BASIC_FONTS.get(ch) {
            let base_x = x + char_index * 8;
            for (row, bits) in glyph.iter().enumerate() {
                for col in 0..8 {
                    if ((bits >> col) & 1) != 0 {
                        let px = base_x + col;
                        let py = y + row;
                        if px < WIDTH && py < HEIGHT {
                            frame[py * WIDTH + px] = color;
                        }
                    }
                }
            }
        }
    }
}

#[allow(dead_code)]
fn _path(value: &str) -> PathBuf {
    PathBuf::from(value)
}
