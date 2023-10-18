import os


reference_file = "impute5/test3/omni4k-10_for_impute5.vcf.gz"
target_file = "impute5/test3/reduced10_for_impute5.vcf.gz"
imputed_file_save_location = "impute5/test3/sandbox/testing_full.vcf.gz"

impute_region = "20:0-80000000"

string = f" impute5/impute5_1.1.5 --h {reference_file} \
    --g {target_file}  --o {imputed_file_save_location} \
    --r {impute_region}"

os.system(string)

