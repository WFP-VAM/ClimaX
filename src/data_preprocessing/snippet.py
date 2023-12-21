import subprocess

# Define the base command
base_cmd = "python src/data_preprocessing/nc2np_equally_ecmwf_downscaling.py --region='Mozambique/Regions/region_{}'"

# Loop through each region number and execute the command
for i in range(1, 10):  # This will loop from 1 to 9
    cmd = base_cmd.format(i)
    subprocess.run(cmd, shell=True)
