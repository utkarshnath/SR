import re
import os
import sys

def parse_logs(filename):
    with open(filename, 'r') as file:
        data = file.read()

    # regex patterns for PSNR, SSIM, LPIPS, and iteration
    psnr_pattern = r"# psnr: (\d+\.\d+)\s+Best: (\d+\.\d+) @ (\d+) iter"
    ssim_pattern = r"# ssim: (\d+\.\d+)\s+Best: (\d+\.\d+) @ (\d+) iter"
    lpips_pattern = r"# lpips: (\d+\.\d+)\s+Best: (\d+\.\d+) @ (\d+) iter"
    iter_pattern = r"\[epoch: \d+, iter: (\d+),"

    # find all matches in the data
    psnr_matches = re.findall(psnr_pattern, data)
    ssim_matches = re.findall(ssim_pattern, data)
    lpips_matches = re.findall(lpips_pattern, data)
    iter_matches = re.findall(iter_pattern, data)
    
    # create a new file to write the output
    base_filename = os.path.splitext(filename)[0]
    output_filename = base_filename + "_output.txt"
    with open(output_filename, 'w') as outfile:
        # check if the lengths match
        for i in range(len(lpips_matches)-1):
            outfile.write('---------------------------------------------\n')
            outfile.write(f"Iteration: {5000*(i+1)}, PSNR: {psnr_matches[i][0]}, SSIM: {ssim_matches[i][0]}, LPIPS: {lpips_matches[i][0]}\n")
            outfile.write('\n')
            outfile.write(f"     Best Iteration: {lpips_matches[i][2]}, PSNR: {psnr_matches[i][1]}, SSIM: {ssim_matches[i][1]}, LPIPS: {lpips_matches[i][1]}\n")


# use the function
parse_logs(sys.argv[1])
