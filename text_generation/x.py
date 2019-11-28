
import numpy as np 

texts = ["I am Peter", "You suck XD"]

indices_list = [np.meshgrid(np.array(i), np.arange(len(text) + 1)) for i, text in enumerate(texts)]

indices_list = np.block(indices_list)
#print(indices_list)


for row in range(indices_list.shape[0]):
    text_index = indices_list[row, 0]
    end_index = indices_list[row, 1]
    
    print(text_index, end_index)
    
