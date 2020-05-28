:: This copies the folder to the Y folder, so that we save the documentation. 
:: /E means we copy folders and subfolders, including Empty folders.
:: /y means we overwrite
:: see https://ss64.com/nt/xcopy.html

xcopy /E/y * Y:\Personal_Learning\my-ds-documentation-master
