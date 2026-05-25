# calculus

## Concept
- Đạo hàm dùng định nghĩa với h cực nhỏ
- Gradient Descent trong cùng một số lượng bước lặp, thì learning rate càng lớn, bước nhảy càng lớn, nếu quá lớn có thể bỏ qua local minimum nhưng khi gặp dốc, có có xu hướng trượt theo hướng xuống tiến đến Local minumum.
- learning rate lớn có khả năng hội tụ nhanh hơn
- Đối với các hàm logarit (khi x tiến về 0 thì dần ra vô cực) và hàm mũ (x tiến ra vô cực thì dần về 0) tồn tại các điểm kì dị tiến ra vô cực, nếu không kiểm soát rất dễ gặp hiện tượng trượt liên tục xuống dốc không có điểm dừng. Vì vậy nên tối ưu hóa bằng giới hạn số bước lặp. 
## Key learnings
- loss function
- gradient descent
- learning rate
- gradient explosion
- singularity (điểm kỳ dị)