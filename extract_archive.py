import os
import zipfile
import rarfile
import sys
import logging
import argparse
from collections import deque, defaultdict
import shutil
import traceback
from tqdm import tqdm  # For progress bar
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("extract_archives.log", encoding='utf-8')
    ]
)

# Argument parsing for verbosity and deletion options
parser = argparse.ArgumentParser(description="Extract zip and rar files.")
parser.add_argument("directory", type=str, help="Directory to process")
parser.add_argument("--verbose", action="store_true", help="Increase output verbosity")
parser.add_argument("--auto-delete", action="store_true", help="Automatically approve all deletions")
parser.add_argument("--unrar-path", type=str, help="Specify the path to UnRAR.exe if different from the default")
parser.add_argument("--dry-run", action="store_true", help="Simulate the extraction process without making changes")
parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of worker threads for parallel extraction")
parser.add_argument("--batch-size", type=int, default=100, help="Number of archives to process per batch")
args = parser.parse_args()

# Set verbosity
if args.verbose:
    logging.getLogger().setLevel(logging.DEBUG)

# Set the path to UnRAR.exe for rarfile module
unrar_path = args.unrar_path if args.unrar_path else r"C:\Program Files\WinRAR\UnRAR.exe"

if os.path.exists(unrar_path) and os.path.isfile(unrar_path):
    rarfile.UNRAR_TOOL = unrar_path
else:
    if not args.unrar_path:
        unrar_path = input("UnRAR tool not found. Please enter the correct path to UnRAR.exe: ")
    if os.path.exists(unrar_path) and os.path.isfile(unrar_path):
        rarfile.UNRAR_TOOL = unrar_path
    else:
        logging.error(f"UnRAR tool still not found. Please ensure it is installed properly.")
        logging.error(f"Visit https://www.win-rar.com/download.html to download and install.")
        input("Press Enter to exit...")
        sys.exit(1)

def has_extracted_files(directory):
    """
    Checks if a directory contains any files, including nested files.
    """
    for root, _, files in os.walk(directory):
        if files:
            return True
    return False

def remove_empty_directories(directory):
    """
    Removes empty directories in the given directory.
    """
    for root, dirs, _ in os.walk(directory, topdown=False):
        for dir_ in dirs:
            dir_path = os.path.join(root, dir_)
            try:
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)
                    logging.info(f"Removed empty directory: {dir_path}")
            except Exception as e:
                logging.error(f"Failed to remove directory {dir_path}: {e}")

def is_zip(file_path):
    """
    Validates if the file is a genuine ZIP archive.
    """
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            bad_file = zip_ref.testzip()
            if bad_file:
                logging.error(f"Corrupted file within ZIP archive: {file_path}, File: {bad_file}")
                return False
            return True
    except zipfile.BadZipFile:
        logging.error(f"Invalid ZIP file format: {file_path}")
        return False
    except Exception as e:
        logging.error(f"Error validating ZIP file {file_path}: {e}")
        return False

def is_rar(file_path):
    """
    Validates if the file is a genuine RAR archive.
    """
    try:
        with rarfile.RarFile(file_path, 'r') as rar_ref:
            # Attempt to test the RAR archive's integrity
            rar_ref.testrar()
            return True
    except rarfile.BadRarFile:
        logging.error(f"Invalid RAR file format: {file_path}")
        return False
    except rarfile.RarCannotExec:
        logging.error(f"Cannot execute UnRAR tool for {file_path}. Check UnRAR path.")
        return False
    except Exception as e:
        logging.error(f"Error validating RAR file {file_path}: {e}")
        return False

# List of specific files to exclude from processing
EXCLUDED_FILES = [
    "Afraid - Nelson, Willie - BFM001 - 47.zip",
    "Charlie Pride - Why Baby Why - CB90065 - 13.zip",
    # Add more filenames as needed
]

# Optionally, you can add patterns or conditions for exclusion
def should_exclude_file(file_name):
    """
    Determines if a file should be excluded based on its name.
    Modify this function to include more complex exclusion logic if needed.
    """
    if file_name in EXCLUDED_FILES:
        return True
    # Example: Exclude files starting with 'temp' or ending with '_backup.zip'
    if file_name.lower().startswith('temp') or file_name.lower().endswith('_backup.zip'):
        return True
    return False

def move_to_failed(file_path, base_dir):
    """
    Moves the problematic archive to a 'failed_archives' directory.
    """
    failed_dir = os.path.join(base_dir, "failed_archives")
    os.makedirs(failed_dir, exist_ok=True)
    new_path = os.path.join(failed_dir, os.path.basename(file_path))
    try:
        shutil.move(file_path, new_path)
        logging.info(f"Moved {file_path} to {new_path}")
    except Exception as e:
        logging.error(f"Failed to move {file_path} to {new_path}: {e}")

def extract_archive(file_path, base_dir, auto_delete=False, dry_run=False):
    """
    Extracts a single archive (zip or rar).
    Returns a list of newly found archives within the extracted contents.
    """
    new_archives = []
    try:
        if file_path.endswith('.zip'):
            if not is_zip(file_path):
                logging.error(f"Skipping invalid ZIP file: {file_path}")
                move_to_failed(file_path, base_dir)
                return new_archives
            if dry_run:
                logging.info(f"Dry run: Would extract ZIP file: {file_path}")
            else:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(os.path.dirname(file_path))
                logging.info(f"Extracted: {file_path}")
        elif file_path.endswith('.rar'):
            if not is_rar(file_path):
                logging.error(f"Skipping invalid RAR file: {file_path}")
                move_to_failed(file_path, base_dir)
                return new_archives
            if dry_run:
                logging.info(f"Dry run: Would extract RAR file: {file_path}")
            else:
                with rarfile.RarFile(file_path, 'r') as rar_ref:
                    rar_ref.extractall(os.path.dirname(file_path))
                logging.info(f"Extracted: {file_path}")
        else:
            logging.warning(f"Unsupported file format for file: {file_path}")
            return new_archives

        # Check if extraction was successful before deleting
        extracted_dir = os.path.dirname(file_path)
        if has_extracted_files(extracted_dir):
            if dry_run:
                logging.info(f"Dry run: Would delete archive: {file_path}")
            else:
                if auto_delete:
                    try:
                        os.remove(file_path)
                        logging.info(f"Deleted: {file_path}")
                    except Exception as e:
                        logging.error(f"Failed to delete {file_path}: {e}")
                else:
                    confirm_delete = input(f"Do you want to delete the archive {file_path}? (y/n/a): ").strip().lower()
                    if confirm_delete == 'y':
                        try:
                            os.remove(file_path)
                            logging.info(f"Deleted: {file_path}")
                        except Exception as e:
                            logging.error(f"Failed to delete {file_path}: {e}")
                    elif confirm_delete == 'a':
                        auto_delete = True
                        try:
                            os.remove(file_path)
                            logging.info(f"Deleted: {file_path}")
                        except Exception as e:
                            logging.error(f"Failed to delete {file_path}: {e}")
                    else:
                        logging.info(f"Skipped deletion of: {file_path}")

            # Scan the extracted directory for nested archives and add them to the new_archives list
            for root, _, files in os.walk(extracted_dir):
                for file in files:
                    if file.endswith('.zip') or file.endswith('.rar'):
                        if should_exclude_file(file):
                            logging.debug(f"Excluded nested archive: {file}")
                            continue
                        nested_file_path = os.path.join(root, file)
                        new_archives.append(nested_file_path)
        else:
            logging.warning(f"No files extracted from {file_path}, skipping deletion.")

    except (zipfile.BadZipFile, rarfile.Error) as e:
        logging.error(f"Failed to extract {file_path}: {e}")
        raise e  # Re-raise exception to handle retries
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
    except PermissionError:
        logging.error(f"Permission denied: {file_path}")
    except Exception as e:
        logging.error(f"Unexpected error with file {file_path}: {e}")
        logging.debug(traceback.format_exc())

    return new_archives

def process_archives_parallel(archives, base_dir, auto_delete, dry_run, retry_counts, max_retries, executor, lock):
    """
    Processes a list of archives in parallel using ThreadPoolExecutor.
    Returns a list of newly found archives.
    """
    new_archives = []
    future_to_file = {}
    for file_path in archives:
        if file_path in retry_counts and retry_counts[file_path] >= max_retries:
            logging.warning(f"Skipping {file_path} as it has reached maximum retry attempts.")
            continue
        future = executor.submit(extract_archive, file_path, base_dir, auto_delete, dry_run)
        future_to_file[future] = file_path

    for future in as_completed(future_to_file):
        file_path = future_to_file[future]
        try:
            result = future.result()
            new_archives.extend(result)
        except Exception as e:
            with lock:
                retry_counts[file_path] += 1
                if retry_counts[file_path] >= max_retries:
                    logging.error(f"Max retries reached for {file_path}. Moving to 'failed_archives' directory.")
                    if not dry_run:
                        move_to_failed(file_path, base_dir)
                else:
                    logging.info(f"Retrying extraction for {file_path} (Attempt {retry_counts[file_path]}/{max_retries})")
                    new_archives.append(file_path)  # Re-add to archives for retry

    return new_archives

def process_directory(directory, auto_delete=False, exclude_dirs=None, dry_run=False, max_workers=4, batch_size=100):
    """
    Traverses all folders in the given directory, extracts zip and rar files in parallel and in batches, and deletes them after extraction.
    Handles nested archives iteratively.
    """
    if exclude_dirs is None:
        exclude_dirs = []

    # Initialize tracking structures
    processed_files = set()
    retry_counts = defaultdict(int)
    lock = threading.Lock()

    # Initial scan for archives
    initial_archives = []
    for root, dirs, files in os.walk(directory):
        # Skip excluded directories
        if any(excluded.lower() in root.lower() for excluded in exclude_dirs):
            logging.debug(f"Excluded directory: {root}")
            continue

        archives_in_directory = [
            os.path.join(root, file) for file in files
            if file.endswith(('.zip', '.rar')) and not should_exclude_file(file)
        ]
        initial_archives.extend(archives_in_directory)

    total_archives = len(initial_archives)
    if total_archives == 0:
        logging.info("No archives found to process.")
        return

    # Initialize progress bar
    pbar = tqdm(total=total_archives, desc="Processing Archives", unit="archive")

    # Initialize ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        current_archives = initial_archives.copy()
        while current_archives:
            # Determine the current batch
            current_batch = current_archives[:batch_size]
            remaining_archives = current_archives[batch_size:]

            # Submit the current batch for parallel processing
            new_archives = process_archives_parallel(
                archives=current_batch,
                base_dir=directory,
                auto_delete=auto_delete,
                dry_run=dry_run,
                retry_counts=retry_counts,
                max_retries=3,
                executor=executor,
                lock=lock
            )

            # Update progress bar
            pbar.update(len(current_batch))

            # Prepare archives for the next iteration
            # Combine remaining_archives with newly found archives
            current_archives = remaining_archives + new_archives

            # Update total in progress bar if new archives are found
            if new_archives:
                pbar.total += len(new_archives)
                pbar.refresh()

    pbar.close()
    remove_empty_directories(directory)

def main():
    base_directory = args.directory
    auto_delete = args.auto_delete
    dry_run = args.dry_run
    max_workers = args.max_workers
    batch_size = args.batch_size

    # Define directories to exclude (case-insensitive)
    exclude_dirs = [
        r"$RECYCLE.BIN",
        r"System Volume Information",
        r"Program Files",
        r"Windows",
        r"AppData"
    ]

    if os.path.isdir(base_directory):
        try:
            process_directory(
                directory=base_directory,
                auto_delete=auto_delete,
                exclude_dirs=exclude_dirs,
                dry_run=dry_run,
                max_workers=max_workers,
                batch_size=batch_size
            )
            logging.info("Processing completed.")
        except Exception as e:
            logging.error(f"An error occurred while processing the directory: {e}")
            logging.debug(traceback.format_exc())
    else:
        logging.error("The specified directory does not exist.")

if __name__ == "__main__":
    main()
