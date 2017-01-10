CC = g++ -std=c++11
Opencv = `pkg-config opencv --cflags --libs`
LNK = -llapack -lblas -larmadillo
all:main.cpp
	$(CC) main.cpp $(LNK) 
clean:
	rm -rf a.out
