import os
import numpy as np

def verify_cohort_topology(cohort_dir: str, expected_vertex_count: int = 10217) -> bool:
    """
    Task C: Validates vertex count consistency and strict index correspondence
    across all compressed .npz matrices within a specific cohort.
    """
    if not os.path.exists(cohort_dir):
        print(f"[ERROR] Cohort directory not found: {cohort_dir}")
        return False

    npz_files = [f for f in os.listdir(cohort_dir) if f.endswith(".npz")]
    if not npz_files:
        print(f"[WARNING] No compressed .npz files found in {cohort_dir}")
        return False

    print(f"\nEvaluating {len(npz_files)} assets inside: {os.path.basename(cohort_dir)}")
    print("-" * 60)

    # Establish the baseline reference framework using the first file
    reference_file = npz_files[0]
    reference_path = os.path.join(cohort_dir, reference_file)
    
    with np.load(reference_path) as ref_data:
        ref_vertices = ref_data["vertices"]
        ref_faces = ref_data["faces"]

    # Initial validation of the reference mesh
    if ref_vertices.shape[0] != expected_vertex_count:
        print(f"[CRITICAL] Reference file {reference_file} failed vertex boundary check.")
        print(f"Expected: {expected_vertex_count}, Found: {ref_vertices.shape[0]}")
        return False

    print(f"[BASELINE] Established reference mesh using: {reference_file}")
    print(f"[BASELINE] Target Topology Matrix Shape: {ref_faces.shape} (Faces)")
    
    mismatched_vertex_count = 0
    scrambled_topology_count = 0
    passed_count = 0

    # Iterate through all remaining files to assert invariant alignment
    for file in npz_files:
        file_path = os.path.join(cohort_dir, file)
        
        with np.load(file_path) as data:
            current_vertices = data["vertices"]
            current_faces = data["faces"]

        # Check 1: Automated Vertex Count Verification
        if current_vertices.shape[0] != expected_vertex_count:
            print(f" -> [FAIL] {file} | Inconsistent Vertex Count: {current_vertices.shape[0]}")
            mismatched_vertex_count += 1
            continue

        # Check 2: Index Correspondence Verification (Face Adjacency Identity)
        # If the topology graph is identical, the face indexing matrices must match perfectly.
        if not np.array_equal(ref_faces, current_faces):
            print(f" -> [FAIL] {file} | Scrambled Index Mapping Detected.")
            scrambled_topology_count += 1
            continue

        passed_count += 1

    # Report synthesis
    print("-" * 60)
    print(f"Validation Summary for {os.path.basename(cohort_dir)}:")
    print(f"  - Successfully Verified (Invariant): {passed_count} / {len(npz_files)}")
    print(f"  - Vertex Count Mismatches:          {mismatched_vertex_count}")
    print(f"  - Scrambled Index Mapping Errors:   {scrambled_topology_count}")
    
    return (mismatched_vertex_count == 0) and (scrambled_topology_count == 0)

if __name__ == "__main__":
    # Absolute PhD workspace directory mapping
    PARSED_DATA_ROOT = r"C:\Users\Vasileios Nikolaou\Documents\PhD\research_pipeline\data_parsed"
    
    female_cohort = os.path.join(PARSED_DATA_ROOT, "female_medium_average")
    male_cohort = os.path.join(PARSED_DATA_ROOT, "male_medium_average")

    print("Executing Phase 2: Topological Invariance Verification Pipeline...")
    
    female_status = verify_cohort_topology(female_cohort, expected_vertex_count=10217)
    male_status = verify_cohort_topology(male_cohort, expected_vertex_count=10217)
    
    print("\n" + "=" * 60)
    if female_status and male_status:
        print("FINAL STATUS: PASSED")
        print("All data cohorts demonstrate perfect structural and topological invariance.")
        print("The matrices are mathematically certified for Graph Neural Network loaders.")
    else:
        print("FINAL STATUS: FAILED")
        print("Topological anomalies detected. Review the failure logs above before training.")
    print("=" * 60)