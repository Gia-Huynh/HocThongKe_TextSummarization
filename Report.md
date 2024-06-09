# Text Summarization
[Blog của Google về Model và Dataset](https://research.google/blog/exploring-transfer-learning-with-t5-the-text-to-text-transfer-transformer/).  
[Trang thông tin chi tiết về Dataset](https://huggingface.co/datasets/billsum).  
## Dataset
Dataset được sử dụng là dataset BillSum, gồm các dự luật của quốc hội Mỹ và bang California (“US Congressional and California state bills”) và tóm tắt của chúng.  
Bao gồm các đặc trưng: Nội dung của Bill, tóm tắt, tiêu đề (chỉ có với dự luật quốc hội, không có với California), độ dài của nội dung, độ dài tóm tắt.  

![](./report_data/BillSumImg.png)  
*Ảnh chụp một phần dữ liệu gốc chưa qua tiền xử lý.*  

## Tiền xử lý dataset  
```
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
```  
Dữ liệu được tokenized theo như báo cáo của mô hình gốc, bao gồm các bước:  
	Một  
	Hai  
	Ba  
	Bốn  
	Năm  
test
## Model  
Model được sử dụng là *google-t5/t5-small* bởi Google.  
Mục tiêu của họ khi tạo ra model này là dùng nó cho nhiều tác vụ khác nhau với một model duy nhất, một metric duy nhất, một thuật toán tối ưu duy nhất nhằm đơn giản hóa quá trình so sánh giữa các bộ dữ liệu và các bài toán khác nhau.
Nó là một model ứng dụng Transformer với input/output đều là văn bản (Khác với model Bart, cũng của Google).  
Tuy nhiên về cái cốt lõi của Transformer thì không có sự thay đổi đáng kể nên nhóm tác giả đã hướng người đọc tới báo cáo gốc của Transformer để hiểu rõ hơn.  
  
"We use a simplified form of position embeddings
where each “embedding” is simply a scalar that is added to the corresponding logit used
for computing the attention weights. For efficiency, we also share the position embedding
parameters across all layers in our model, though within a given layer each attention head
uses a different learned position embedding"
"To summarize, our model is roughly equivalent to the original Transformer proposed by
Vaswani et al. (2017) with the exception of removing the Layer Norm bias, placing the layer
normalization outside the residual path, and using a different position embedding scheme.
Since these architectural changes are orthogonal to the experimental factors we consider in
our empirical survey of transfer learning, we leave the ablation of their impact for future
work."
  
Đây là hỏi chatGpt để đọc, mà thấy không đáng tin tí nào cả nên thôi chắc xóa phần này rồi đọc paper cho chắc  
"Certainly! The architecture of the Text-to-Text Transfer Transformer (T5) model is based on the Transformer architecture, which has been highly successful in natural language processing (NLP) tasks. Here are the key components:

Encoder-Decoder Structure:
T5 follows the standard Transformer architecture, consisting of an encoder and a decoder.
The encoder processes the input text, while the decoder generates the output text.
Both encoder and decoder consist of multiple layers of self-attention and feed-forward neural networks.
Unified Text-to-Text Format:
T5 reframes all NLP tasks into a consistent format: both input and output are treated as text strings.
This allows T5 to handle a wide range of tasks without task-specific modifications.
Pre-training and Fine-tuning:
T5 is pre-trained on a large corpus of text data using unsupervised learning.
During pre-training, it learns to predict masked tokens (similar to BERT) and also learns to generate text.
Fine-tuning involves training T5 on specific downstream tasks with labeled data.
Task-Specific Heads:
T5 uses task-specific heads (output layers) for different tasks.
For example, a translation task would have a different head than a summarization task.
Shared Vocabulary:
T5 uses a shared vocabulary for both input and output tokens.
This simplifies the model and allows it to handle diverse tasks.
Task Prefixes:
T5 introduces task-specific prefixes to guide the model during fine-tuning.
For instance, a translation task might have the prefix “Translate English to French: …”
Overall, T5’s strength lies in its versatility—it can be fine-tuned for various NLP tasks by simply adjusting the task-specific head and providing appropriate prefixes during fine-tuning. 😊🚀

If you have any more questions or need further details, feel free to ask!"
  
  
### Pre-train dataset
Model đã được pre-train trên tập dataset *Colossal Clean Crawled Corpus (C4)*, vốn là một tập dữ liệu văn bản khổng lồ được crawl từ dữ liệu khắp nơi trên internet.  
Nhóm Google thấy tập đó có số lượng lớn nhưng chất lượng cực thấp nên họ đã tiến hành "lọc" lại bộ dữ liệu như sau:  
"""
• Chỉ giữ những câu kết thúc bằng dấu câu. (Là: "? . !")  
• Loại bỏ các trang web ít hơn 3 câu, chỉ giữ những câu dài hơn 4 chữ.  
• Loại bỏ các trang có những từ "nhạy cảm".
• Nhiều trang đưa ra cảnh báo cần bật Javascript nên họ loại bỏ những câu nào có chữ "Javascript".
• Loại bỏ các trang có cụm từ "Lorem Ipsum" vì nó là văn mẫu để tạm.
• Một số trang web sẽ phải chứa code. Vì ngoặc nhọn "{" được dùng trong nhiều ngôn ngữ lập trình (như Javascript, vốn được dùng rất nhiều trên web),  loại bỏ trang nào chứa ngoặc nhọn.  
• Loại bỏ các ký hiệu dẫn nguồn (Như của Wikipedia là "[1], [Cần dẫn chứng],...") nên loại bỏ hết các ký hiệu đó.  
• Many pages had boilerplate policy notices, so we removed any lines containing the
strings “terms of use”, “privacy policy”, “cookie policy”, “uses cookies”, “use of
cookies”, or “use cookies”.
• Để giảm sự trùng lặp cho bộ dữ liệu, tiến hành loại bỏ các bộ 3 câu liên tiếp bị trùng trong cả bộ dữ liệu, giữ lại đúng một bộ.
## Metric  
### Rouge score 
* Là nhóm các độ đo được dùng để đánh giá chất lượng của text summarization. Cách mà Rough score hoạt động là so sánh văn bản tóm tắt được tạo ra và văn bản tóm tắt tham chiếu.
#### a.	Rouge-N: 
-	Đo lường **tần suất xuất hiện của n-gram** (cụm từ n từ liên tiếp) trong tóm tắt máy so với tóm tắt tham chiếu.
-	Trong rouge-n, n có thể là 1 hoặc 2. 
-	Ví dụ: “I love her very much”. Rouge-1: ( I ) , ( love ), ( her ), ( very ), ( much ).Rouge-2: ( I love ), ( love her ), ( her very ), ( very much )
-	Cách tính cụ thể của Rouge 1 và Rouge 2:
Machine result: “the cat is on the mat”. 
Reference result: “the cat sat on the red mat”.
-	**Rouge-1:** 
+ Liệt kê các unigrams:
Machine result: ["the", "cat", "is", "on", "the", "mat"] => 6
Reference result: ["the", "cat", "sat", "on", "the", “red”, "mat"] => 7
+ Đếm số lượng unigrams phù hợp giữ machine với reference:
 	["the", "cat",  "on", "the", "mat"]   => 5
+ Tính precision ( P ), recall ( R ) và f1-score ( F1 )
**P = số unigrams phù hợp / tổng số unigrams trong machine result.** P= 5/6 = 0.833.
**R = số unigrams phù hợp / tổng số unigrams trong reference result.** R = 5/7 = 0.714.
**F1 = (2xPxR) / (P+R)**. F1 = (2x0.833x0.714)/(0.833+0.714) = 0.769
-	**Rouge-2:**
+ Liệt kê các unigrams:
Machine result: ["the cat", "cat is", "is on", "on the", "the mat"] => 5
Reference result: ["the cat", "cat sat","sat on","on the", "the red",“red mat”] => 6
+ Đếm số lượng unigrams phù hợp giữ machine với reference:
 	["the cat", "on the"]   => 2
+ Tính precision ( P ), recall ( R ) và f1-score ( F1 )
**P = số unigrams phù hợp / tổng số unigrams trong machine result.** P = 2/5 = 0.4
**R = số unigrams phù hợp / tổng số unigrams trong reference result.** R = 2/6 = 0.333
**F1 = (2xPxR) / (P+R).** F1 = (2x0.4x0.333)/(0.4+0.333) = 0.364
#### b. Rouge-L:
-	Đo lường **chiều dài chuỗi con chung dài nhất** (Longest Common Subsequence - LCS) giữa tóm tắt máy và tóm tắt tham chiếu. Thay vì chỉ quan tâm đến các n-gram cụ thể, ROUGE-L xem xét các chuỗi con dài nhất mà cả hai câu cùng có và đánh giá mức độ trùng khớp về trật tự từ.
-	Tính 3 chỉ số: Precision ( P ), Recall ( R ), F1-score ( F1 )
+ P: tỷ lệ độ dài của LCS với tổng số từ trong machine result. 
   P = |LCS| / length of machine result.
+ R: tỷ lệ độ dài của LCS với tổng số từ trong reference result.
   R = |LCS| / length of reference result.
+ F1 = (2xPxR) / (P+R)
-	Ví dụ: 
 Machine result: “the cute cat is on the mat”. => 7.
 Reference result: “the cute cat sat on the red mat”. => 8
+ Tìm chuỗi con giống nhau dài nhất: “the cute cat” => len = 3.
+ Tính P, R, F1:
P = 3/7 = 0.429
R = 3/8 = 0.375
F1 = (2xPxR) / (P+R) = 0.281
#### c.	Rouge-S:
-	Đo **dựa trên các cặp từ** (bigram) mà không yêu cầu các từ trong cặp phải liền kề nhau nhưng vẫn giữ nguyên thứ tự xuất hiện. ROUGE-S đánh giá khả năng của tóm tắt máy trong việc bảo tồn các mối quan hệ giữa các từ trong tóm tắt tham chiếu.
-	Tính 3 chỉ số: P, R, F1
+ Precision (P): Tỷ lệ skip-bigram phù hợp so với tổng số skip-bigram trong tóm tắt máy.
+ Recall (R): Tỷ lệ skip-bigram phù hợp so với tổng số skip-bigram trong tóm tắt tham chiếu.
+ F1 = (2xPxR) / (P+R)
-	Ví dụ: Machine result : "the cat is on the mat" Reference result: "the cat sat on the red mat"
+ Liệt kê các skip-bigrams trong machine result và reference result.
Skip-bigrams trong machine result: "the cat", "the is", "the on",”the the”, "the mat", "cat is", "cat on", “cat the”, "cat mat", "is on", “is the”, "is mat", "on the", "on mat", “the mat” => 15
Skip-bigrams trong tóm tắt tham chiếu: "the cat", "the sat", "the on", “the the”, "the red", "the mat", "cat sat", "cat on", “cat the”, "cat red", "cat mat", "sat on", “sat the”, "sat red", "sat mat", "on the", "on red", "on mat", "the red", "the mat", "red mat" => 21
+ So sánh các skip-bigrams trong tóm tắt máy với các skip-bigrams trong tóm tắt tham chiếu để tìm các cặp trùng khớp:
"the cat", "the on", "the the", "the mat", "cat on", "cat the", "cat mat", "on the", "on mat”.  => 9
+ Tính Precision, Recall và F1-Score
P = 9/15 = 0.6
R = 9/21 = 0.429
F1 = (2x0.6x0.429) / (0.6+0.429) = 0.5





## Kết quả huấn luyện  
Khá tốt, so sánh trực tiếp với kết quả của model có sẵn thì cho thấy nó output giống nhau.  
Cho thấy quá trình huấn luyện đã thành công. 
![](./report_data/OutputDemo.png)  
*Ảnh chụp so sánh giữa output của mô hình huấn luyện được (3) và mô hình có sẵn (2)*  

