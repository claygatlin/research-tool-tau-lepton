"""Example runner for the TAV MC Validation Suite.

This script imports `run_full_validation` from the integrated module and
prints the function signature and a short usage hint. It will not run the
full validation unless valid ROOT files are provided.
"""

from inspect import signature


def main():
    try:
        from menus.particle.cms.tav_mc_validation_suite import run_full_validation
    except Exception as e:
        print(f"Import failed: {e}")
        return

    sig = signature(run_full_validation)
    print("run_full_validation is available with signature:")
    print(sig)
    print("\nTo run the validation, call run_full_validation(...) with real ROOT paths.")
    print("Example (edit paths and run):\n")
    print('''from menus.particle.cms.tav_mc_validation_suite import run_full_validation
validation_results = run_full_validation(
    data_file="/path/to/Run2012BC_DoubleMuParked_Muons.root",
    mc_files=["/path/to/DYJetsToLL.root","/path/to/TTbar.root"],
    output_dir="tav_mc_validation_full",
    pileup_weight_dict=None,
    do_pileup_reweight=True,
    do_dependence_studies=True,
    verbose=True
)
print(validation_results["summary"])''')


if __name__ == "__main__":
    main()
