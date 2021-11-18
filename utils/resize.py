from PIL import Image
import os, sys

def resize_image(image_dir):

    dirs = [name for name in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, name))]
    
    for dir in dirs:
    
        dir_name = image_dir + '\\' + dir
        save_dir = dir_name + '_resized'
        
        if dir_name.split('_')[-1] == 'resized': continue
        
        if not os.path.exists(save_dir): 
            os.makedirs(save_dir)
            
        else: continue
        
        print(dir_name.split('\\')[-1])

        images = [x for x in os.listdir(dir_name) if x.endswith(".jpg")]
        
        for image in images:
            
            file_dir = dir_name + '\\' + image
            file = Image.open(file_dir)
            
            width, height = file.size
            if width > height:
                left = (width - height) / 2
                right = width - left
                top = 0
                bottom = height
            else:
                top = (height - width) / 2
                bottom = height - top
                left = 0
                right = width
                
            file = file.crop((left, top, right, bottom))
            file = file.resize([224, 224], Image.ANTIALIAS)
            file.save(os.path.join(save_dir, image), file.format)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        "Please give a video as an argument to this program"
    else:
        resize_image(sys.argv[1])