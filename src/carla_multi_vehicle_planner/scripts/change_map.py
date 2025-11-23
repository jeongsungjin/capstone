# import carla


# client = carla.Client("localhost", 2000)
# client.set_timeout(10.0)
# world = client.get_world()

# world = client.load_world('bigmap')
import carla
c = carla.Client('localhost', 2000); c.set_timeout(5.0)
w = c.get_world(); print(w.get_map().name)
print(len(w.get_map().to_opendrive()) > 0)  # True 여야 정상