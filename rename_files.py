# importing os module
import os
  
# Function to rename multiple files
def main():

    directory_name = "data/review_data/train/neg/test"
    for count, filename in enumerate(os.listdir(directory_name)):
        filename_wo_ext = filename.split('.')[0]
        rating = int(filename_wo_ext.split('_')[1])
        file_index = int(filename_wo_ext.split('_')[0])
        new_file_index = file_index + 5000
        new_filename = f'{new_file_index}_{rating}.txt'
        
        src = directory_name + '/' + filename
        dst = directory_name + '/' + new_filename
          
        # rename() function will
        # rename all the files
        os.rename(src, dst)
  
# Driver Code
if __name__ == '__main__':
      
    # Calling main() function
    main()