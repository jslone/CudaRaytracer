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
    lastTick = SDL_GetTicks();
    
    while(running) {
      int32_t startTick = lastTick;

      // Do stuff
      handle_events();
      
      // Sleep to avoid going over frameRate
      int32_t period = 1000 / frameRate;
      int32_t endTick = SDL_GetTicks();
      int32_t sleepTime = period - (endTick - startTick);

      lastTick = endTick;
      SDL_Delay(math::max(sleepTime,1));
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
