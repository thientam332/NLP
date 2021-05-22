import wordcloud
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
cloud = np.array(X_Train).flatten()
plt.figure(figsize=(20,10))
word_cloud = wordcloud.WordCloud(max_words=100,background_color ="black",
                               width=2000,height=1000,mode="RGB").generate(str(cloud))
plt.axis("off")
plt.imshow(word_cloud)
# Kiểm tra tấn suất xuất hiện của 1 từ, từ đó xuất hiện càng nhiều thì kích cỡ chữ càng lớn
