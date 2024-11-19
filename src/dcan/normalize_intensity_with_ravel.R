normalize_intensity_with_ravel <- function(scan_path, output_dir) {
  
  # author: "Paul Reiners"
  # date: "2024-11-13"

  library(fslr)
  if (have.fsl()) {
    print(fsl_version())
  }
  base_name <- basename(scan_path)
  len <- nchar(base_name)
  n_last <- 7 
  base_name <- substr(base_name, 1, nchar(base_name) - n_last)
  print(paste("base_name:", base_name))

  scan_reg_n4_brain = readNIfTI(scan_path)

  ortho2(scan_reg_n4_brain, crosshairs=FALSE, mfrow=c(1, 3), add.orient=FALSE, ylim=c(0, 400))
  
  scan_reg_n4_brain_seg <- fast(scan_reg_n4_brain, verbose=FALSE, opts="-t 1 -n 3") 
  ortho2(scan_reg_n4_brain_seg, crosshairs=FALSE, mfrow=c(1, 3), add.orient=FALSE)

  output_file_path = file.path(output_dir, paste(base_name, "segmentation_result", sep="_"))
  print(paste("output_file_path:", output_file_path))
  writeNIfTI(scan_reg_n4_brain_seg, filename = output_file_path)
}
