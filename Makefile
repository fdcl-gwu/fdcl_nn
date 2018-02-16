INCLUDE_PATH= ./eigen-3.3.4
CFLAGS=$(foreach d, $(INCLUDE_PATH), -I$d) -Wall -std=c++11
VRPN_LIBS = -lvrpn -lquat -pthread
OPTI_FLAG = -O3
GTK_LIBS = -rdynamic `pkg-config --cflags gtk+-3.0`  `pkg-config --libs gtk+-3.0` 

mlp: c_mlp.o mlp.o c_softmax.o
	g++ -o mlp mlp.o c_mlp.o c_softmax.o -lm $(CFLAGS)

mlp.o: MLP.cpp 
	g++ -c MLP.cpp $(CFLAGS)

c_mlp.o: c_mlp.cpp c_mlp.h
	g++ -c c_mlp.cpp  $(CFLAGS)

c_softmax.o: c_softmax.cpp c_softmax.h
	g++ -c c_softmax.cpp  $(CFLAGS)
	
