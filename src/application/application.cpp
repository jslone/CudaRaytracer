#include <iostream>
#include "application.h"

namespace acr {

  Application::Application(const Args args)
    : renderer(args.renderer),
      frameRate(args.frameRate) {
    
      SDL_Init(0);
      SDL_InitSubSystem(SDL_INIT_TIMER | SDL_INIT_EVENTS);
      atexit(SDL_Quit);
  }

  Application::~Application() {
  }

  void Application::start() {
    running = true;
    run();
  }

  void Application::quit() {
    running = false;
  }

  void Application::run() {
    int32_t lagTime = 0;
    lastTick = SDL_GetTicks();
    
    while(running) {
      int32_t startTime = SDL_GetTicks(); // lastTick;

      // Do stuff
      handle_events();
      
      // Sleep to avoid going over frameRate
      int32_t period = 1000 / frameRate;
      int32_t endTime = SDL_GetTicks();
      int32_t deltaTime = endTime - startTime;
      int32_t sleepTime = period - deltaTime - lagTime;
      int32_t wakeTime = endTime + sleepTime;
      
      //std::cout << "deltaTime: " << endTime - lastTick << std::endl;

      // Update last tick so we know when this frame ended
      lastTick = endTime;
      
      // Sleep until period is up
      do {
        SDL_Delay(math::max(sleepTime,1));
      } while((int)(SDL_GetTicks()) < wakeTime);
      
      // Update lag time
      lagTime = sleepTime > 0 ? SDL_GetTicks() - wakeTime : -sleepTime;
    }
  }

  void Application::handle_events() {
    SDL_Event event;
    while(SDL_PollEvent(&event)) {
      switch(event.type) {
        case SDL_QUIT:
          quit();
          break;
        default:
          break;
      }
    }
  }

} // namespace acr


int main(int argc, char **argv) {
  // Setup
  acr::Application::Args args;
  args.renderer.title = "CudaRenderer";
  args.renderer.pos.x = 0;
  args.renderer.pos.y = 0;
  args.renderer.dim.x = 800;
  args.renderer.dim.y = 600;
  args.frameRate = 20;
  
  // Start the app
  acr::Application app(args);
  app.start();
  
  return 0;
}
