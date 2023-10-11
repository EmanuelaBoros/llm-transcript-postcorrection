import os
import tarfile
import glob

def gather_and_archive_files(model_names, folder_path, output_folder):
    """
    Gather files and create archives based on model names.

    Parameters:
        model_names (list of str): List of model names to gather and archive files.
        folder_path (str): Path to the folder containing the files.
        output_folder (str): Path to the folder where archives will be saved.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    for run_type in ['few-shot', 'zero-shot']:

        for prompt in ['prompt_basic_01', 'prompt_basic_02', 'prompt_complex_01', 'prompt_complex_02',
                       'prompt_complex_03_per_lang']:
            # Loop through all model names
            for model_name in model_names:
                # Use glob to find all files that start with the model name
                files_to_archive = glob.glob(os.path.join(folder_path, f"{model_name}_{run_type}_{prompt}*"))

                # Check if any files were found
                if files_to_archive:
                    # Create an archive for the found files
                    archive_name = os.path.join(output_folder, f"{model_name}_{run_type}_{prompt}.tar.gz")

                    with tarfile.open(archive_name, "w:gz") as tar:
                        # Add files into the tar file
                        for file_path in files_to_archive:
                            tar.add(file_path, arcname=os.path.basename(file_path))

                    print(f"Archive created: {archive_name}")
                else:
                    print(f"No files found starting with {model_name}.")

                # Delete the files after adding them to the archive
                for file_path in files_to_archive:
                    os.remove(file_path)
# Example usage:
model_names = ["ajmc-mixed", "ajmc-primary-text", "htrec", "icdar-2017", "icdar-2019",
               "impresso-nzz", "ina", "overproof"]
gather_and_archive_files(model_names, "./processed_data", "./output_archives")

