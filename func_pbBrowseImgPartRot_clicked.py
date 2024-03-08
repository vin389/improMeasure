def func_pbBrowseImgPartRot_clicked(self):
    pass 
    Rot_Img_path = QFileDialog.getExistingDirectory(self, "Directory of source files", "./")
    self.ui.edImgPartRot.setText(Rot_Img_path)
