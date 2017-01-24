CC = g++ -std=c++11 -O3
Opencv = `pkg-config opencv --cflags --libs`
LNK = -llapack -lblas -larmadillo
Output = -o AE.o
all:main.cpp
	$(CC) main.cpp $(LNK) $(Opencv) $(Output)
clean:
	rm -rf $(Output)
