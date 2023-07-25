from utils import FeaturePoint, FeatureTrack

track_1 = FeatureTrack()
track_1.push(FeaturePoint(0, 0))
print(repr(track_1), " Length: %d"%track_1.length)
track_1.push(FeaturePoint(0, 1))
print(repr(track_1), " Length: %d"%track_1.length)
track_1.push(FeaturePoint(0, 2))
print(repr(track_1), " Length: %d"%track_1.length)
track_1.push(FeaturePoint(0, 3))
print(repr(track_1), " Length: %d"%track_1.length)
track_1.push(FeaturePoint(0, 4))
print(repr(track_1), " Length: %d"%track_1.length)
track_1.delete(track_1.latest.prev.prev)
print(repr(track_1), " Length: %d"%track_1.length)
track_1.delete_latest()
print(repr(track_1), " Length: %d"%track_1.length)
track_1.delete_oldest()
print(repr(track_1), " Length: %d"%track_1.length)
track_1.delete_oldest()
print(repr(track_1), " Length: %d"%track_1.length)
track_1.delete_oldest()
print(repr(track_1), " Length: %d"%track_1.length)
track_1.delete_oldest()
print(repr(track_1), " Length: %d"%track_1.length)
track_1.delete_latest()
print(repr(track_1), " Length: %d"%track_1.length)
track_1.push(FeaturePoint(0, 5))
print(repr(track_1), " Length: %d"%track_1.length)
