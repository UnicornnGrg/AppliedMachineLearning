"""
Master Pipeline Script for Feature Engineering
==============================================
Purpose: Sequentially executes the entire feature engineering pipeline:
         1. Statistical Feature Selection
         2. Business Logic Feature Selection
         3. Realistic Feature Selection
         4. PCA on Statistical Features
         5. PCA on Business Features
         6. PCA on Realistic Features

Usage: Run this script to regenerate all feature sets and PCA reductions.
"""

import subprocess
import sys
import os
import time

def run_script(script_name):
    """Run a python script and check for errors."""
    print(f"\n{'='*80}")
    print(f"RUNNING: {script_name}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    # Get the directory of the current script to ensure correct paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, script_name)
    
    if not os.path.exists(script_path):
        print(f"❌ ERROR: Script not found: {script_path}")
        return False
    
    try:
        # Run the script using the current python interpreter
        # cwd=current_dir ensures imports like 'import fe_utils' work correctly
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            cwd=current_dir
        )
        
        duration = time.time() - start_time
        print(f"\n✅ SUCCESS: {script_name} completed in {duration:.2f} seconds.")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR: {script_name} failed with exit code {e.returncode}.")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: Failed to run {script_name}. Reason: {str(e)}")
        return False

def main():
    print("\n" + "="*80)
    print("STARTING FEATURE ENGINEERING PIPELINE")
    print("="*80)
    
    # Define the execution order
    scripts = [
        # Phase 1: Feature Selection & Engineering
        "01_fe_statistical.py",
        "02_fe_business.py",
        "03_fe_realistic.py",
        
        # Phase 2: Dimensionality Reduction (PCA)
        # These depend on the outputs of Phase 1
        "04_fe_pca_statistical.py",
        "05_fe_pca_business.py",
        "06_fe_pca_realistic.py"
    ]
    
    total_start = time.time()
    success_count = 0
    
    for script in scripts:
        if run_script(script):
            success_count += 1
        else:
            print("\n⚠️ Pipeline stopped due to error in previous step.")
            break
            
    total_duration = time.time() - total_start
    
    print("\n" + "="*80)
    print("PIPELINE SUMMARY")
    print("="*80)
    print(f"Total Scripts: {len(scripts)}")
    print(f"Successful:    {success_count}")
    print(f"Failed:        {len(scripts) - success_count}")
    print(f"Total Time:    {total_duration:.2f} seconds")
    print("="*80)
    
    if success_count == len(scripts):
        print("\n✨ All feature engineering tasks completed successfully!")
    else:
        print("\n⚠️ Some tasks failed. Please check the logs above.")

if __name__ == "__main__":
    main()
