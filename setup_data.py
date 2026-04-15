import os
import subprocess
import zipfile

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory at: {os.path.abspath(data_dir)}")
    record_id = "8265464"
    print(f"Downloading Zenodo Record {record_id}...")
    subprocess.run(["python", "-m", "zenodo_get", record_id, "-o", data_dir])
    print("Extracting files...")
    for item in os.listdir(data_dir):
        if item.endswith(".zip"):
            file_path = os.path.join(data_dir, item)
            print(f"  Unzipping {item}...")
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            os.remove(file_path)
            print(f"Finished {item}")
    print(f"Dataset ready in: {os.path.abspath(data_dir)}")

if __name__ == "__main__":
    main()