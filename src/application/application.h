#include "renderer/renderer.h"

namespace acr {

/* Application class.
 *
 * Application implements a class to manage each submodule of the
 * entire application. Application is in charge of managing the
 * entry point and the per frame execution until the application
 * terminates.
 */
class Application {
public:
  Application();
  ~Application();
private:
  Renderer renderer;
};

} // namespace acr
