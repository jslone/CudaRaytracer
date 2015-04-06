#include <iostream>
#include "application.h"

void Application::init() {
	std::cout << "hi" << std::endl;
}

void Application::uninit() {
	std::cout << "bye" << std::endl;
}

int main(int argc, char **argv) {
	Application app;
	app.init();
	app.uninit();
}