#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PROFILE="full"
if [[ "${1:-}" == "--host-ci" ]]; then
  PROFILE="host-ci"
  shift
elif [[ "${1:-}" == "--offline" ]]; then
  PROFILE="offline"
  shift
elif [[ "${1:-}" == "--quick" ]]; then
  PROFILE="quick"
  shift
elif [[ "${1:-}" == "--full" ]]; then
  PROFILE="full"
  shift
elif [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'USAGE'
Usage: scripts/verify.sh [--host-ci|--offline|--quick|--full]

Runs the ROCm-Oxide production verification gate.

  --host-ci  Host-only GitHub CI gate; does not build the root ROCm crate.
  --offline  ROCm-installed docs, clippy, package, and tool-crate tests.
  --quick    Unit/tool tests plus core GPU smoke coverage.
  --full     Full local gate, including heavier examples and visual artifacts.

Artifacts are written under target/production-readiness/.
Set ROCM_OXIDE_VERIFY_TIMEOUT=0 to disable the default 1200s per-command timeout.
USAGE
  exit 0
fi

if [[ "$#" -ne 0 ]]; then
  echo "unknown verify argument: $1" >&2
  echo "run scripts/verify.sh --help for usage" >&2
  exit 2
fi

ARTIFACT_DIR="$ROOT/target/production-readiness"
LOG="$ARTIFACT_DIR/verify-${PROFILE}.log"
VERIFY_TIMEOUT="${ROCM_OXIDE_VERIFY_TIMEOUT:-1200s}"
mkdir -p "$ARTIFACT_DIR"
: > "$LOG"

run() {
  printf '\n$'
  printf ' %q' "$@"
  printf '\n'
  {
    printf '\n$'
    printf ' %q' "$@"
    printf '\n'
  } >> "$LOG"
  if [[ -n "$VERIFY_TIMEOUT" && "$VERIFY_TIMEOUT" != "0" ]] && command -v timeout >/dev/null 2>&1; then
    timeout --foreground "$VERIFY_TIMEOUT" "$@" 2>&1 | tee -a "$LOG"
  else
    "$@" 2>&1 | tee -a "$LOG"
  fi
}

run_device_example() {
  local example="$1"
  shift
  run cargo run --features device-spike --example "$example" -- "$@"
}

run_demo() {
  local manifest="$1"
  shift
  run cargo run --manifest-path "$manifest" -- "$@"
}

run_demo_bin() {
  local manifest="$1"
  local bin="$2"
  shift 2
  run cargo run --manifest-path "$manifest" --bin "$bin" -- "$@"
}

audit_artifacts() {
  local validation_json="$ARTIFACT_DIR/validation_profile.json"
  local performance_json="$ARTIFACT_DIR/performance_probe.json"

  printf '\n$ python3 <artifact-audit> %q %q %q %q\n' "$PROFILE" "$ARTIFACT_DIR" "$validation_json" "$performance_json"
  {
    printf '\n$ python3 <artifact-audit> %q %q %q %q\n' "$PROFILE" "$ARTIFACT_DIR" "$validation_json" "$performance_json"
  } >> "$LOG"

  python3 - "$PROFILE" "$ARTIFACT_DIR" "$validation_json" "$performance_json" <<'PY' 2>&1 | tee -a "$LOG"
import hashlib
import json
import math
import sys
from pathlib import Path

profile = sys.argv[1]
artifact_dir = Path(sys.argv[2])
validation_path = Path(sys.argv[3])
performance_path = Path(sys.argv[4])
errors = []


def fail(message):
    errors.append(message)


def load_json(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        fail(f"missing artifact: {path}")
    except json.JSONDecodeError as err:
        fail(f"invalid JSON artifact {path}: {err}")
    except OSError as err:
        fail(f"could not read artifact {path}: {err}")
    return None


def require_file(path, label):
    if not path.is_file():
        fail(f"missing {label}: {path}")
        return False
    if path.stat().st_size == 0:
        fail(f"empty {label}: {path}")
        return False
    return True


def require_png(path):
    if not require_file(path, "PNG artifact"):
        return
    try:
        header = path.read_bytes()[:8]
    except OSError as err:
        fail(f"could not read PNG artifact {path}: {err}")
        return
    if header != b"\x89PNG\r\n\x1a\n":
        fail(f"artifact is not a PNG file: {path}")


def file_sha256(path):
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as err:
        fail(f"could not hash artifact {path}: {err}")
        return None
    return digest.hexdigest()


def artifact_record(path):
    path = Path(path)
    digest = file_sha256(path)
    if digest is None:
        raise RuntimeError(f"could not hash {path}")
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": digest,
    }


def manifest_path_for_metadata(path):
    if path.name.endswith(".metadata.json"):
        return path.with_name(path.name[:-len(".metadata.json")] + ".manifest.json")
    return path.with_suffix(".manifest.json")


def require_manifest_artifact(record, label, expected_path=None):
    if not isinstance(record, dict):
        fail(f"release manifest {label} artifact is not an object")
        return None
    raw_path = non_empty_string(record.get("path"), f"release manifest {label}.path")
    if raw_path is None:
        return None
    path = Path(raw_path)
    if expected_path is not None:
        try:
            if path.resolve() != Path(expected_path).resolve():
                fail(
                    f"release manifest {label}.path {path} does not match expected {expected_path}"
                )
        except OSError as err:
            fail(f"could not resolve release manifest {label}.path {path}: {err}")
    if not require_file(path, f"release manifest {label} artifact"):
        return path
    size = record.get("size")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        fail(f"release manifest {label}.size must be a positive integer")
    elif path.stat().st_size != size:
        fail(
            f"release manifest {label}.size {size} does not match actual size {path.stat().st_size}"
        )
    sha256 = non_empty_string(record.get("sha256"), f"release manifest {label}.sha256")
    if sha256 is not None:
        actual = file_sha256(path)
        if actual is not None and actual != sha256:
            fail(f"release manifest {label}.sha256 {sha256} does not match actual {actual}")
    return path


def finite_positive(value, label):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        fail(f"{label} is not numeric: {value!r}")
        return
    if not math.isfinite(value) or value <= 0:
        fail(f"{label} must be finite and positive: {value!r}")


def non_empty_string(value, label):
    if not isinstance(value, str) or not value:
        fail(f"{label} must be a non-empty string")
        return None
    return value


RESOURCE_KEYS = [
    "kernarg_segment_size",
    "kernarg_segment_align",
    "max_flat_workgroup_size",
    "group_segment_fixed_size",
    "private_segment_fixed_size",
    "sgpr_count",
    "vgpr_count",
    "sgpr_spill_count",
    "vgpr_spill_count",
    "wavefront_size",
    "uses_dynamic_shared_mem",
    "uses_dynamic_stack",
]

validation = load_json(validation_path)
performance = load_json(performance_path)

arch = None
if validation is not None:
    if validation.get("format") != "rocm-oxide-validation-profile-v1":
        fail(f"{validation_path} has unexpected format {validation.get('format')!r}")
    arch = non_empty_string(validation.get("selected_arch"), "validation selected_arch")
    if arch is not None and not arch.startswith("gfx"):
        fail(f"validation selected_arch must name a ROCm GPU target: {arch!r}")
    known_profile = validation.get("known_profile")
    if not isinstance(known_profile, dict):
        fail("validation profile is missing known_profile object")
    else:
        if arch is not None and known_profile.get("arch") != arch:
            fail(
                "validation known_profile.arch "
                f"{known_profile.get('arch')!r} does not match selected_arch {arch!r}"
            )
        if known_profile.get("known") is not True:
            fail(f"validation profile for {arch!r} is not a known release baseline")
        deviations = known_profile.get("deviations")
        if not isinstance(deviations, list):
            fail("validation known_profile.deviations must be a list")
        elif deviations:
            fail(f"validation profile has known-profile deviations: {deviations!r}")
    if not isinstance(validation.get("skipped_tests"), list):
        fail("validation skipped_tests must be a list")

metadata = None
metadata_path = None
manifest = None
manifest_path = None
if performance is not None:
    if performance.get("format") != "rocm-oxide-performance-probe-v2":
        fail(f"{performance_path} has unexpected format {performance.get('format')!r}")
    perf_arch = non_empty_string(performance.get("arch"), "performance arch")
    if arch is not None and perf_arch != arch:
        fail(f"performance arch {perf_arch!r} does not match validation arch {arch!r}")
    metadata_raw = non_empty_string(performance.get("metadata"), "performance metadata path")
    if metadata_raw is not None:
        metadata_path = Path(metadata_raw)
        if require_file(metadata_path, "generated metadata"):
            metadata = load_json(metadata_path)
            manifest_path = manifest_path_for_metadata(metadata_path)
            if require_file(manifest_path, "generated release manifest"):
                manifest = load_json(manifest_path)
    samples = performance.get("samples")
    if not isinstance(samples, list) or not samples:
        fail("performance samples must be a non-empty list")

metadata_by_kernel = {}
if metadata is not None:
    if metadata.get("target") != "amdgcn-amd-amdhsa":
        fail(f"metadata target must be amdgcn-amd-amdhsa, got {metadata.get('target')!r}")
    if arch is not None and metadata.get("arch") != arch:
        fail(f"metadata arch {metadata.get('arch')!r} does not match validation arch {arch!r}")

    hsaco_raw = non_empty_string(metadata.get("hsaco"), "metadata hsaco path")
    if hsaco_raw is not None:
        require_file(Path(hsaco_raw), "generated HSACO")

    link = metadata.get("link")
    objects = link.get("objects") if isinstance(link, dict) else None
    if not isinstance(objects, list) or not objects:
        fail("metadata link.objects must be a non-empty list")
        objects = []

    link_kernel_names = []
    for index, obj in enumerate(objects):
        if not isinstance(obj, dict):
            fail(f"metadata link.objects[{index}] is not an object")
            continue
        non_empty_string(obj.get("package"), f"metadata link.objects[{index}].package")
        for field in ["llvm_ir", "object"]:
            raw_path = non_empty_string(obj.get(field), f"metadata link.objects[{index}].{field}")
            if raw_path is not None:
                require_file(Path(raw_path), f"metadata link object {field}")
        kernels = obj.get("kernels")
        if not isinstance(kernels, list) or not kernels:
            fail(f"metadata link.objects[{index}].kernels must be a non-empty list")
        else:
            for kernel in kernels:
                name = non_empty_string(kernel, f"metadata link.objects[{index}].kernels[]")
                if name is not None:
                    link_kernel_names.append(name)

    kernels = metadata.get("kernels")
    if not isinstance(kernels, list) or not kernels:
        fail("metadata kernels must be a non-empty list")
        kernels = []

    seen = set()
    for index, kernel in enumerate(kernels):
        if not isinstance(kernel, dict):
            fail(f"metadata kernels[{index}] is not an object")
            continue
        name = non_empty_string(kernel.get("name"), f"metadata kernels[{index}].name")
        if name is None:
            continue
        if name in seen:
            fail(f"duplicate metadata kernel entry: {name}")
        seen.add(name)
        code_object = kernel.get("code_object")
        if not isinstance(code_object, dict):
            fail(f"metadata kernel {name} is missing code_object resource facts")
            continue
        metadata_by_kernel[name] = code_object
        for key in RESOURCE_KEYS:
            if key not in code_object:
                fail(f"metadata kernel {name} is missing code_object.{key}")
        if code_object.get("uses_dynamic_stack") is True:
            fail(f"metadata kernel {name} uses dynamic stack")
        if code_object.get("sgpr_spill_count", 0) not in (None, 0):
            fail(f"metadata kernel {name} has SGPR spills: {code_object.get('sgpr_spill_count')}")
        if code_object.get("vgpr_spill_count", 0) not in (None, 0):
            fail(f"metadata kernel {name} has VGPR spills: {code_object.get('vgpr_spill_count')}")

    if set(link_kernel_names) != set(metadata_by_kernel):
        missing_from_link = sorted(set(metadata_by_kernel) - set(link_kernel_names))
        missing_from_metadata = sorted(set(link_kernel_names) - set(metadata_by_kernel))
        fail(
            "metadata link-object kernels do not match generated kernel metadata; "
            f"missing_from_link={missing_from_link}, missing_from_metadata={missing_from_metadata}"
        )

if manifest is not None:
    if manifest.get("format") != "rocm-oxide-release-manifest-v1":
        fail(f"{manifest_path} has unexpected format {manifest.get('format')!r}")
    if manifest.get("target") != "amdgcn-amd-amdhsa":
        fail(f"release manifest target must be amdgcn-amd-amdhsa, got {manifest.get('target')!r}")
    if arch is not None and manifest.get("arch") != arch:
        fail(f"release manifest arch {manifest.get('arch')!r} does not match validation arch {arch!r}")

    generated_epoch_ms = manifest.get("generated_epoch_ms")
    if not isinstance(generated_epoch_ms, int) or isinstance(generated_epoch_ms, bool) or generated_epoch_ms <= 0:
        fail("release manifest generated_epoch_ms must be a positive integer")

    tools = manifest.get("tools")
    if not isinstance(tools, dict):
        fail("release manifest is missing tools object")
    else:
        for key in ["cargo", "rustc", "llc", "clang", "llvm_readelf", "llvm_objdump"]:
            tool = tools.get(key)
            if not isinstance(tool, dict):
                fail(f"release manifest tools.{key} is not an object")
                continue
            non_empty_string(tool.get("path"), f"release manifest tools.{key}.path")
            non_empty_string(tool.get("source"), f"release manifest tools.{key}.source")
            non_empty_string(tool.get("version"), f"release manifest tools.{key}.version")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        fail("release manifest is missing artifacts object")
    else:
        metadata_hsaco = metadata.get("hsaco") if isinstance(metadata, dict) else None
        if metadata_hsaco is not None:
            require_manifest_artifact(artifacts.get("hsaco"), "artifacts.hsaco", Path(metadata_hsaco))
        else:
            require_manifest_artifact(artifacts.get("hsaco"), "artifacts.hsaco")
        if metadata_hsaco is not None:
            compiler_metadata = Path(metadata_hsaco).with_suffix(".metadata.json")
            require_manifest_artifact(artifacts.get("metadata"), "artifacts.metadata", compiler_metadata)
        else:
            require_manifest_artifact(artifacts.get("metadata"), "artifacts.metadata")
        require_manifest_artifact(artifacts.get("bindings"), "artifacts.bindings")

    manifest_link = manifest.get("link")
    manifest_objects = manifest_link.get("objects") if isinstance(manifest_link, dict) else None
    if not isinstance(manifest_objects, list) or not manifest_objects:
        fail("release manifest link.objects must be a non-empty list")
        manifest_objects = []
    metadata_objects = []
    if isinstance(metadata, dict) and isinstance(metadata.get("link"), dict):
        metadata_objects = metadata["link"].get("objects") or []
    if len(manifest_objects) != len(metadata_objects):
        fail(
            f"release manifest link object count {len(manifest_objects)} "
            f"does not match metadata link object count {len(metadata_objects)}"
        )
    for index, obj in enumerate(manifest_objects):
        if not isinstance(obj, dict):
            fail(f"release manifest link.objects[{index}] is not an object")
            continue
        metadata_obj = metadata_objects[index] if index < len(metadata_objects) and isinstance(metadata_objects[index], dict) else {}
        package = non_empty_string(obj.get("package"), f"release manifest link.objects[{index}].package")
        if metadata_obj and package != metadata_obj.get("package"):
            fail(
                f"release manifest link.objects[{index}].package {package!r} "
                f"does not match metadata package {metadata_obj.get('package')!r}"
            )
        require_manifest_artifact(obj.get("llvm_ir"), f"link.objects[{index}].llvm_ir", metadata_obj.get("llvm_ir"))
        require_manifest_artifact(obj.get("object"), f"link.objects[{index}].object", metadata_obj.get("object"))
        kernels = obj.get("kernels")
        metadata_kernels = metadata_obj.get("kernels") if metadata_obj else None
        if not isinstance(kernels, list) or not kernels:
            fail(f"release manifest link.objects[{index}].kernels must be a non-empty list")
        elif metadata_kernels is not None and kernels != metadata_kernels:
            fail(
                f"release manifest link.objects[{index}].kernels {kernels!r} "
                f"does not match metadata kernels {metadata_kernels!r}"
            )

    manifest_kernels = manifest.get("kernels")
    if not isinstance(manifest_kernels, list) or not manifest_kernels:
        fail("release manifest kernels must be a non-empty list")
    else:
        manifest_by_kernel = {}
        for index, kernel in enumerate(manifest_kernels):
            if not isinstance(kernel, dict):
                fail(f"release manifest kernels[{index}] is not an object")
                continue
            name = non_empty_string(kernel.get("name"), f"release manifest kernels[{index}].name")
            if name is None:
                continue
            if name in manifest_by_kernel:
                fail(f"duplicate release manifest kernel entry: {name}")
            resources = kernel.get("resources")
            if not isinstance(resources, dict):
                fail(f"release manifest kernel {name} is missing resources object")
                continue
            manifest_by_kernel[name] = resources
            metadata_resource = metadata_by_kernel.get(name)
            if metadata_resource is None:
                fail(f"release manifest kernel {name} is missing from generated metadata")
                continue
            for key in RESOURCE_KEYS:
                if resources.get(key) != metadata_resource.get(key):
                    fail(
                        f"release manifest kernel {name} resource {key}={resources.get(key)!r} "
                        f"does not match metadata {metadata_resource.get(key)!r}"
                    )
        if metadata_by_kernel and set(manifest_by_kernel) != set(metadata_by_kernel):
            fail(
                "release manifest kernels do not match metadata kernels; "
                f"manifest_only={sorted(set(manifest_by_kernel) - set(metadata_by_kernel))}, "
                f"metadata_only={sorted(set(metadata_by_kernel) - set(manifest_by_kernel))}"
            )

if performance is not None and isinstance(performance.get("samples"), list):
    for index, sample in enumerate(performance["samples"]):
        if not isinstance(sample, dict):
            fail(f"performance samples[{index}] is not an object")
            continue
        kernel_name = non_empty_string(sample.get("kernel"), f"performance samples[{index}].kernel")
        finite_positive(sample.get("gpu_ms"), f"performance sample {kernel_name or index} gpu_ms")
        finite_positive(sample.get("est_fps"), f"performance sample {kernel_name or index} est_fps")
        resources = sample.get("resources")
        if not isinstance(resources, dict):
            fail(f"performance sample {kernel_name or index} is missing resource facts")
            continue
        if kernel_name is not None and resources.get("name") != kernel_name:
            fail(
                f"performance resource name {resources.get('name')!r} "
                f"does not match sample kernel {kernel_name!r}"
            )
        if resources.get("uses_dynamic_stack") is True:
            fail(f"performance sample {kernel_name} uses dynamic stack")
        if resources.get("sgpr_spill_count", 0) not in (None, 0):
            fail(f"performance sample {kernel_name} has SGPR spills: {resources.get('sgpr_spill_count')}")
        if resources.get("vgpr_spill_count", 0) not in (None, 0):
            fail(f"performance sample {kernel_name} has VGPR spills: {resources.get('vgpr_spill_count')}")

        occupancy = sample.get("occupancy")
        if not isinstance(occupancy, dict):
            fail(f"performance sample {kernel_name or index} is missing occupancy facts")
        else:
            finite_positive(
                occupancy.get("block_threads"),
                f"performance sample {kernel_name or index} occupancy.block_threads",
            )
            finite_positive(
                occupancy.get("active_blocks_per_multiprocessor"),
                f"performance sample {kernel_name or index} occupancy.active_blocks_per_multiprocessor",
            )

        if kernel_name is not None and metadata_by_kernel:
            metadata_resource = metadata_by_kernel.get(kernel_name)
            if metadata_resource is None:
                fail(f"performance sample {kernel_name} is missing from generated metadata")
            else:
                for key in RESOURCE_KEYS:
                    if resources.get(key) != metadata_resource.get(key):
                        fail(
                            f"performance sample {kernel_name} resource {key}={resources.get(key)!r} "
                            f"does not match metadata {metadata_resource.get(key)!r}"
                        )

require_png(artifact_dir / "spectral_lattice_chain.png")
if profile == "full":
    for name in [
        "spectral_lattice.png",
        "spectral_lattice_core.png",
        "spectral_lattice_lds.png",
        "spectral_lattice_atomic.png",
        "spectral_lattice_4k.png",
    ]:
        require_png(artifact_dir / name)

if errors:
    print("artifact audit failed:")
    for error in errors:
        print(f"- {error}")
    sys.exit(1)


def production_visual_artifacts():
    names = ["spectral_lattice_chain.png"]
    if profile == "full":
        names.extend(
            [
                "spectral_lattice.png",
                "spectral_lattice_core.png",
                "spectral_lattice_lds.png",
                "spectral_lattice_atomic.png",
                "spectral_lattice_4k.png",
            ]
        )
    return [{"name": name, **artifact_record(artifact_dir / name)} for name in names]


try:
    production_manifest_path = artifact_dir / "release_manifest.json"
    production_manifest = {
        "format": "rocm-oxide-production-readiness-manifest-v1",
        "profile": profile,
        "target": manifest.get("target"),
        "arch": arch,
        "toolchain": manifest.get("tools"),
        "compiler": {
            "manifest": artifact_record(manifest_path),
            "artifact_set": manifest,
        },
        "validation": {
            "artifact": artifact_record(validation_path),
            "format": validation.get("format"),
            "selected_arch": validation.get("selected_arch"),
            "known_profile": validation.get("known_profile"),
            "skipped_test_count": len(validation.get("skipped_tests", [])),
        },
        "performance": {
            "artifact": artifact_record(performance_path),
            "format": performance.get("format"),
            "metadata": performance.get("metadata"),
            "sample_count": len(performance.get("samples", [])),
        },
        "visual_artifacts": production_visual_artifacts(),
    }
    production_manifest_path.write_text(
        json.dumps(production_manifest, indent=2, sort_keys=True) + "\n"
    )
except Exception as err:
    print("artifact audit failed:")
    print(f"- could not write production-readiness manifest: {err}")
    sys.exit(1)

sample_count = 0
if isinstance(performance, dict) and isinstance(performance.get("samples"), list):
    sample_count = len(performance["samples"])
kernel_count = len(metadata_by_kernel)
print(
    "artifact audit passed: "
    f"profile={profile}, arch={arch}, samples={sample_count}, metadata_kernels={kernel_count}, "
    f"manifest={production_manifest_path}"
)
PY
}

run cargo test --manifest-path crates/rocm-oxide-kernel/Cargo.toml -- --test-threads=1
run cargo test --manifest-path tools/rocm-oxide-build/Cargo.toml -- --test-threads=1
run cargo test --manifest-path tools/cargo-rocm-oxide/Cargo.toml -- --test-threads=1

run_host_ci_checks() {
  run cargo fmt --check
  run bash -n scripts/verify.sh
  run bash -n scripts/consumer-smoke.sh
  run cargo package --allow-dirty --no-verify
}

if [[ "$PROFILE" == "host-ci" ]]; then
  run_host_ci_checks
  printf '\nverification profile `%s` passed; artifacts: %s\n' "$PROFILE" "$ARTIFACT_DIR"
  exit 0
fi

if [[ "$PROFILE" == "offline" ]]; then
  run_host_ci_checks
  run cargo doc --no-deps
  run cargo clippy --all-targets -- -D warnings -A clippy::too_many_arguments
  printf '\nverification profile `%s` passed; artifacts: %s\n' "$PROFILE" "$ARTIFACT_DIR"
  exit 0
fi

run python3 --version
run cargo test -- --test-threads=1
run cargo run --manifest-path tools/cargo-rocm-oxide/Cargo.toml -- rocm-oxide doctor
run cargo run --manifest-path tools/cargo-rocm-oxide/Cargo.toml -- rocm-oxide pipeline
run cargo run --example vector_add
run_device_example rust_device_generated_bindings
run_device_example feature_showcase
run scripts/consumer-smoke.sh
run cargo run --example validation_profile -- --json "$ARTIFACT_DIR/validation_profile.json"
run_device_example performance_probe --json "$ARTIFACT_DIR/performance_probe.json"
run_demo demo-projects/spectral-lattice/Cargo.toml --frames 1 --mode chain --output "$ARTIFACT_DIR/spectral_lattice_chain.png"

if [[ "$PROFILE" == "full" ]]; then
  run_device_example rust_device_add_one
  run_device_example rust_device_vector_add
  run_demo demo-projects/compiler-feature-lab/Cargo.toml --frames 1
  run cargo run --example pinned_stream_vector_add
  run cargo run --example device_operation_chain
  run cargo run --example module_global
  run_demo_bin demo-projects/upscale-artifacts/Cargo.toml depth_aware_upscale
  run_demo_bin demo-projects/upscale-artifacts/Cargo.toml temporal_upscale
  run_demo demo-projects/bvh-raytrace-benchmark/Cargo.toml
  run_demo demo-projects/spectral-lattice/Cargo.toml --frames 3 --output "$ARTIFACT_DIR/spectral_lattice.png"
  for mode in core lds atomic chain; do
    run_demo demo-projects/spectral-lattice/Cargo.toml --frames 3 --mode "$mode" --output "$ARTIFACT_DIR/spectral_lattice_${mode}.png"
  done
  run_demo demo-projects/spectral-lattice/Cargo.toml --frames 1 --mode chain --resolution 4k --fps-limit 120 --gpu-work 256 --output "$ARTIFACT_DIR/spectral_lattice_4k.png"
  run cargo run
fi

audit_artifacts

printf '\nverification profile `%s` passed; artifacts: %s\n' "$PROFILE" "$ARTIFACT_DIR"
