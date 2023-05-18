import os
import csv

root_dir = 'dataset/Audio'  # Root directory where the bird name folders are located
csv_file = 'metadata/bird_dataset_full.csv'  # Name of the CSV file to be generated
bird_index_file = 'metadata/bird_index.csv'  # Name of the CSV file to store bird name and corresponding index

bird_to_index = {}
index = 0

# Open the CSV file in write mode
with open(csv_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Bird Name', 'Audio File Name', 'Path', 'Bird Index'])  # Write the header row

    # Iterate through each bird name folder
    for bird_folder in os.listdir(root_dir):
        bird_path = os.path.join(root_dir, bird_folder)
        print(bird_path)

        # Check if the item in the bird_folder path is a directory
        if os.path.isdir(bird_path):
            # Check if bird is already assigned an index, if not assign a new index
            if bird_folder not in bird_to_index:
                bird_to_index[bird_folder] = index
                index += 1

            # Iterate through each xav file in the bird name folder
            for audio_file in os.listdir(bird_path):
                print(audio_file)
                audio_path = os.path.join(bird_folder, audio_file)
                writer.writerow([bird_folder, audio_file, audio_path, bird_to_index[bird_folder]])

# Save bird to index mapping to a new CSV file
with open(bird_index_file, 'w', newline='') as indexfile:
    writer = csv.writer(indexfile)
    writer.writerow(['Bird Name', 'Bird Index'])  # Write the header row
    for bird, index in bird_to_index.items():
        writer.writerow([bird, index])