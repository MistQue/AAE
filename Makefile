CC = g++ -std=c++11
Opencv = `pkg-config opencv --cflags --libs`
LNK = -llapack -lblas -larmadillo -O2
Output = -o nn.o
all:main.cpp
	$(CC) main.cpp $(LNK) $(Opencv) $(Output)
clean:
	rm -rf $(Output)
