from feature_manager import FeaturePoint, FeatureTrack, FrameNode, FrameTrack, FeatureManager

# Test FeatureTrack
ft_track_1 = FeatureTrack(0)
ft_track_1.push(FeaturePoint(0, 0))
print(repr(ft_track_1), " feature track length %d"%ft_track_1.length)
ft_track_1.push(FeaturePoint(0, 1))
print(repr(ft_track_1), " feature track length %d"%ft_track_1.length)
ft_track_1.push(FeaturePoint(0, 2))
print(repr(ft_track_1), " feature track length %d"%ft_track_1.length)
ft_track_1.push(FeaturePoint(0, 3))
print(repr(ft_track_1), " feature track length %d"%ft_track_1.length)
ft_track_1.pop()
print(repr(ft_track_1), " feature track length %d"%ft_track_1.length)
ft_track_1.pop()
print(repr(ft_track_1), " feature track length %d"%ft_track_1.length)
ft_track_1.pop()
print(repr(ft_track_1), " feature track length %d"%ft_track_1.length)
ft_track_1.pop()
print(repr(ft_track_1), " feature track length %d"%ft_track_1.length)
ft_track_1.pop()
print(repr(ft_track_1), " feature track length %d"%ft_track_1.length)
ft_track_1.push(FeaturePoint(0, 5))
print(repr(ft_track_1), " feature track length %d"%ft_track_1.length)

# Test FrameTrack
frame_track = FrameTrack()
frame_1 = FrameNode(2)
for i in range(4):
    frame_1.add_pt(FeaturePoint(i, 2))
frame_2 = FrameNode(3)
for i in range(2, 5, 1):
    frame_2.add_pt(FeaturePoint(i, 3))

frame_track.push(frame_1)
print(repr(frame_track), "Frame track length: %d"%frame_track.length)
frame_track.push(frame_2)
print(repr(frame_track), "Frame track length: %d"%frame_track.length)

# Test FeatureManager
feature_manager = FeatureManager()
hist = [[0,1,2,3],
          [0,1,2,3],
          [2,3,4],
          [4,5]]
for i, l in enumerate(hist):
    feature_points = [FeaturePoint(track_id=k, timestamp=i) for k in l]
    feature_manager.add_feature_points(feature_points)
    print(repr(feature_manager.frame_track))
    for k, track in feature_manager.feature_tracks.items():
        print(repr(track))