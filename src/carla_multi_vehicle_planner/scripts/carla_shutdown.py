#!/usr/bin/env python3

import rospy

try:
    # Ensure CARLA API path if helper exists (side-effect insert)
    from setup_carla_path import CARLA_BUILD_PATH  # noqa: F401
except Exception:
    pass

try:
    import carla
except ImportError as exc:
    raise RuntimeError(f"CARLA import failed in carla_shutdown: {exc}")


class CarlaShutdown:
    def __init__(self):
        rospy.init_node("carla_shutdown", anonymous=False)

        self.reset_world_on_shutdown = bool(rospy.get_param("~reset_world_on_shutdown", True))

        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        try:
            self.world = self.client.get_world()
        except Exception:
            self.world = None

        rospy.on_shutdown(self._on_shutdown)
        rospy.loginfo("carla_shutdown: ready (reset_world_on_shutdown=%s)", str(self.reset_world_on_shutdown))

    def _on_shutdown(self):
        if not self.reset_world_on_shutdown:
            return
        try:
            current_map = None
            try:
                if self.world is not None:
                    current_map = self.world.get_map().name
            except RuntimeError:
                current_map = None
            self.client.reload_world()
            if current_map:
                try:
                    self.client.load_world(current_map)
                except RuntimeError:
                    pass
            rospy.loginfo("carla_shutdown: CARLA world reloaded on shutdown")
        except RuntimeError as exc:
            rospy.logwarn("carla_shutdown: failed to reload CARLA world on shutdown: %s", exc)


def main():
    try:
        CarlaShutdown()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()


