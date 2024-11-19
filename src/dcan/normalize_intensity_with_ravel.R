normalize_intensity_with_ravel <- function(scan_path, output_dir) {
  # Author: Paul Reiners
  # Date: 2024-11-13
  
  library(fslr)
  
  # Check if FSL is available
  if (have.fsl()) {
    cat("FSL Version:", fsl_version(), "\n")
  } else {
    stop("FSL is not installed or configured properly.")
  }
  
  # Extract base name without the last 7 characters (e.g., file extension)
  base_name <- tools::file_path_sans_ext(basename(scan_path))
  base_name <- substr(base_name, 1, nchar(base_name) - 7)
  cat("Base name:", base_name, "\n")
  
  # Read the NIfTI scan
  scan_reg_n4_brain <- readNIfTI(scan_path)
  
  # Visualize the original brain scan
  ortho2(scan_reg_n4_brain, crosshairs = FALSE, mfrow = c(1, 3), add.orient = FALSE, ylim = c(0, 400))
  
  # Apply FAST segmentation with 3 tissue types
  scan_reg_n4_brain_seg <- fast(scan_reg_n4_brain, verbose = FALSE, opts = "-t 1 -n 3")
  
  # Visualize the segmented brain
  ortho2(scan_reg_n4_brain_seg, crosshairs = FALSE, mfrow = c(1, 3), add.orient = FALSE)
  
  # Construct output file path
  output_file_path <- file.path(output_dir, paste0(base_name, "_segmentation_result"))
  cat("Output file path:", output_file_path, "\n")
  
  # Save the segmented scan
  writeNIfTI(scan_reg_n4_brain_seg, filename = output_file_path)
}
