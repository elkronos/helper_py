#!/usr/bin/env python3
import os
import shutil
import argparse
import logging

def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="For each file in the specified folder, create a folder (named after the file's base name) and move the file into it."
    )
    parser.add_argument("path", help="Path to the folder containing files")
    parser.add_argument("--dry-run", action="store_true", help="Simulate the operations without making any changes")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--recursive", action="store_true", help="Process files in subdirectories recursively")
    return parser.parse_args()

def setup_logging(verbose):
    """
    Setup logging level.
    """
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(message)s")

def get_files(folder_path, recursive=False):
    """
    Retrieve a list of files from the folder.
    If recursive is True, it traverses subdirectories.
    
    Returns:
        List[str]: List of file paths.
    """
    files = []
    if recursive:
        for root, _, filenames in os.walk(folder_path):
            for filename in filenames:
                files.append(os.path.join(root, filename))
    else:
        files = [os.path.join(folder_path, f)
                 for f in os.listdir(folder_path)
                 if os.path.isfile(os.path.join(folder_path, f))]
    return files

def create_folder_for_file(file_path, dry_run=False):
    """
    Create a folder named after the file's base name (without extension) in the same directory.
    
    Returns:
        str: The target folder path.
    """
    folder_path = os.path.dirname(file_path)
    base_name, _ = os.path.splitext(os.path.basename(file_path))
    target_folder = os.path.join(folder_path, base_name)
    
    if not os.path.exists(target_folder):
        logging.info(f"Creating folder: {target_folder}")
        if not dry_run:
            os.makedirs(target_folder)
    return target_folder

def resolve_collision(destination_path):
    """
    If the destination file exists, append a counter suffix to the filename.
    
    Returns:
        str: A unique destination file path.
    """
    if not os.path.exists(destination_path):
        return destination_path
    
    base, ext = os.path.splitext(destination_path)
    counter = 1
    new_destination = f"{base}_{counter}{ext}"
    while os.path.exists(new_destination):
        counter += 1
        new_destination = f"{base}_{counter}{ext}"
    return new_destination

def move_file(file_path, target_folder, dry_run=False):
    """
    Move the file to the target folder.
    Handles file name collisions by renaming if necessary.
    """
    file_name = os.path.basename(file_path)
    destination_path = os.path.join(target_folder, file_name)
    destination_path = resolve_collision(destination_path)
    
    logging.info(f"Moving '{file_path}' to '{destination_path}'")
    if not dry_run:
        shutil.move(file_path, destination_path)

def organize_files(folder_path, dry_run=False, recursive=False):
    """
    Organize files: for each file, create a folder and move the file into it.
    """
    files = get_files(folder_path, recursive=recursive)
    
    if not files:
        logging.info("No files found to process.")
        return

    for file_path in files:
        try:
            target_folder = create_folder_for_file(file_path, dry_run=dry_run)
            move_file(file_path, target_folder, dry_run=dry_run)
        except Exception as e:
            logging.error(f"Error processing file '{file_path}': {e}")

def main():
    args = parse_arguments()
    setup_logging(args.verbose)
    
    if not os.path.isdir(args.path):
        logging.error(f"Error: '{args.path}' is not a valid directory.")
        return

    organize_files(args.path, dry_run=args.dry_run, recursive=args.recursive)

if __name__ == "__main__":
    main()
