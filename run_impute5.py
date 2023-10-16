import os


reference_file = "impute5/test2/testing.vcf.gz"
target_file = "impute5/test2/target.bcf"
imputed_file_save_location = "impute5/test2/imputed.bcf"

impute_region = "20:1000000-4000000"

string = f" impute5/impute5_1.1.5 --h {reference_file} \
    --g {target_file}  --o {imputed_file_save_location} \
    --r {impute_region}"

os.system(string)

