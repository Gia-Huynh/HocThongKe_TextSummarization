# Text Summarization
[Blog của Google về Model và Dataset](https://research.google/blog/exploring-transfer-learning-with-t5-the-text-to-text-transfer-transformer/).  
[Trang HuggingFace về Model](https://huggingface.co/docs/transformers/en/model_doc/t5).
[Trang thông tin chi tiết về Dataset](https://huggingface.co/datasets/billsum).
## Model  
$\quad$ Model được sử dụng là *google-t5/t5-small* bởi Google.  
$\quad$ Mục tiêu của họ khi tạo ra model này là dùng nó cho nhiều tác vụ khác nhau với một model duy nhất, một metric duy nhất, một thuật toán tối ưu duy nhất nhằm đơn giản hóa quá trình so sánh giữa các bộ dữ liệu và các bài toán khác nhau.
$\quad$ Nó là một model ứng dụng Transformer với input là văn bản, output là phiên bản mã hóa của văn bản, theo cấu trúc Encoder - Decoder (Khác với model Bart, cũng của Google). Tuy nhiên về cái cốt lõi của Transformer thì không có sự thay đổi đáng kể nên nhóm tác giả đã hướng người đọc tới báo cáo gốc của Transformer để hiểu rõ hơn.  
$\quad$ Nguyên văn của họ (đã dịch):  
$\quad$ "Chúng tôi đơn giản hóa những "nhúng vị trí" (position embedding), trong đó mỗi “nhúng” chỉ đơn giản là một con số được đưa vào hàm logit (ln (p/(1-p))) tương ứng mà được sử dụng để tính toán trọng số của lớp attention. Để tối ưu hiệu năng, chúng tôi cũng chia sẻ các tham số "nhúng vị trí" đó cho tất cả các lớp trong mô hình, mặc dù trong mỗi lớp, các đầu attention sử dụng tham số học được khác nhau đối với các "nhúng" đó."  
$\quad$ "Ngắn gọn thì mô hình của chúng tôi **gần như** tương đồng với mô hình Transformer gốc được đề xuất bởi Vaswani và cộng sự (2017) với các sự khác biệt là: Loại bỏ bias Layer Norm, đặt lớp chuẩn hóa theo lớp (Layer Normalization) ngoài lớp nối tắt, và sử dụng khác phương pháp nhúng vị trí. Vì sự thay đổi cấu trúc này không ảnh hưởng tới các yếu tố mà chúng tôi đang nghiên cứu (Nghiên cứu về quá trình học chuyển giao), sự ảnh hưởng của việc thay đổi cấu trúc cứ để tương lai rồi tính".
  
### Pre-train dataset
$\quad$ Trước khi huấn luyện trên các tập dataset chất lượng cao dành riêng cho bài toán này, đa phần các mô hình sẽ được pre-train sẵn trên các tập dataset văn bản không gán nhãn để *tập cho mô hình xuất ra được văn bản gần giống tiếng người*.  
$\quad$ Model đã được pre-train trên tập dataset *Colossal Clean Crawled Corpus (C4)*, vốn là một tập dữ liệu văn bản khổng lồ được crawl từ dữ liệu khắp nơi trên internet.  
$\quad$ Nhóm Google thấy tập đó có số lượng lớn nhưng chất lượng cực thấp nên họ đã tiến hành "lọc" lại bộ dữ liệu như sau:  
* Chỉ giữ những câu kết thúc bằng dấu câu. (Là: "? . !")  
* Loại bỏ các trang web ít hơn 3 câu, chỉ giữ những câu dài hơn 4 chữ.  
* Loại bỏ các trang có những từ "nhạy cảm".
* Nhiều trang đưa ra cảnh báo cần bật Javascript nên họ loại bỏ những câu nào có chữ "Javascript".
* Loại bỏ các trang có cụm từ "Lorem Ipsum" vì nó là văn mẫu để tạm.
* Một số trang web sẽ phải chứa code. Vì ngoặc nhọn "{" được dùng trong nhiều ngôn ngữ lập trình (như Javascript, vốn được dùng rất nhiều trên web),  loại bỏ trang nào chứa ngoặc nhọn.  
* Loại bỏ các ký hiệu dẫn nguồn (Như của Wikipedia là "[1], [Cần dẫn chứng],...") nên loại bỏ hết các ký hiệu đó.  
* Many pages had boilerplate policy notices, so we removed any lines containing the
strings “terms of use”, “privacy policy”, “cookie policy”, “uses cookies”, “use of
cookies”, or “use cookies”.
• Để giảm sự trùng lặp cho bộ dữ liệu, tiến hành loại bỏ các bộ 3 câu liên tiếp bị trùng trong cả bộ dữ liệu, giữ lại đúng một bộ.
  
## Dataset
$\quad$ Dataset được sử dụng là dataset BillSum, gồm các dự luật của quốc hội Mỹ và bang California (“US Congressional and California state bills”) và tóm tắt của chúng.  
$\quad$ Bao gồm các đặc trưng: Nội dung của Bill, tóm tắt, tiêu đề (chỉ có với dự luật quốc hội, không có với California), độ dài của nội dung, độ dài tóm tắt.  

![](./report_data/BillSumImg.png)  
*Ảnh chụp một phần dữ liệu gốc chưa qua tiền xử lý.*  

### Tiền xử lý dataset 
```
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
```  
Dữ liệu được tokenized theo như báo cáo của mô hình T5, vì mô hình này có cấu trúc tương tự với Transformer nên quá trình tiền xử lý dataset cũng tương tự báo cáo gốc, bao gồm các bước:  
1. Gán thêm câu khởi tạo tác vụ, ở đây là "Summarize: ". 
2. Mã hóa văn bản đầu vào thành vector các token, ở đây không biến đổi thông qua các mô hình Word2Vec mà chỉ ánh xạ trực tiếp các từ sang một giá trị index của từ trong từ điển, đưa nó vào cột "Input_Id". 
3. Tạo cột "Attention_Mask" để gán nhãn Attention cho mô hình, 1 là quan tâm, 0 là mặc kệ, dùng cho trường hợp mình cần thêm padding vào các câu ngắn cho độ dài đồng đều.  
4. Tạo cột "Labels" chứa phiên bản đã mã hóa (như bước 2) của câu mục tiêu, ở đây là câu tóm tắt.  
  
![](./report_data/TokenizerDemo.png)  
*Ảnh file tokenizer.json minh họa cho bước 1 và 2* 

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

