import numpy as np
from numpy.lib import recfunctions as rfn
import cv2

# Structure array data type for frame 
feature_dtype = np.dtype([('pix', 'f8', (2)),
                          ('pos', 'f8', (3)),
                          ('track_id', 'int32'),
                          ('desc', 'f8', (128))])

class Node:
    def __init__(self):
        self.next = None
        self.prev = None

class DoublyLinkedList:
    def __init__(self) -> None:
        self.latest = None
        self.oldest = None
        self.length = 0

    # Adding a new node to latest		
    def push(self, new_node: Node):
        if self.length == 0:
            self.latest = new_node
            self.oldest = new_node
            self.length = 1
            return
        
        new_node.prev = self.latest
        self.latest.next = new_node
        self.latest = new_node
        self.length += 1

    # Remove the oldest node
    def pop(self):
        if self.length <= 1:
            self.clear()
        else:
            self.oldest = self.oldest.next
            self.oldest.prev = None
            self.length -= 1

    def clear(self):
        self.latest = None
        self.oldest = None
        self.length = 0

    def delete_latest(self):
        if self.length <= 1:
            self.clear()
        else:
            self.latest = self.latest.prev
            self.latest.next = None
            self.length -= 1

    def delete_oldest(self):
        if self.length <= 1:
            self.clear()
        else:
            self.oldest = self.oldest.next
            self.oldest.prev = None
            self.length -= 1


class FeaturePoint(Node):
    def __init__(self, track_id, timestamp, pos=None, pix=None, desc=None):
        super().__init__()
        self.track_id = track_id
        self.timestamp = timestamp
        self.pos = pos
        self.pix = pix
        self.desc = desc

    def __repr__(self):
        return "%d %.2f"%(self.track_id, self.timestamp)

class FeatureTrack(DoublyLinkedList):
    def __init__(self, track_id):
        super().__init__()
        self.track_id = track_id

    def __repr__(self):
        node = self.latest
        nodes = []
        while node is not None:
            nodes.append(repr(node))
            node = node.prev
        nodes.append("None")
        return " -> ".join(nodes)

class FrameNode(Node):
    def __init__(self, timestamp: float, rgb: np.array, depth: np.array):
        super().__init__()
        self.timestamp = timestamp
        self.rgb = rgb
        self.depth = depth
        self.gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        self.pts = []

    def add_pt(self, point_feature: FeaturePoint):        
        pt = np.zeros(1, dtype=feature_dtype)
        pt['pix'] = point_feature.pix
        pt['pos'] = point_feature.pos
        pt['track_id'] = point_feature.track_id
        pt['desc'] = point_feature.desc
        self.pts.append(pt)

    def get_pts_info(self):
        pts = rfn.stack_arrays(self.pts)
        track_ids, pos, pix = pts['track_id'], pts['pos'], pts['pix']
        track_ids = np.ma.getdata(track_ids)
        pos = np.ma.getdata(pos)
        pix = np.ma.getdata(pix)
        return track_ids, pos, pix

    def __repr__(self):
        ids, _, _ = self.get_pts_info()
        ids = ['%d'%i for i in ids]
        return " ".join(ids)

class FrameTrack(DoublyLinkedList):
    def __init__(self):
        super().__init__()

    # Adding a new frame to latest		
    def push(self, new_node: FrameNode):
        if self.length == 0:
            self.latest = new_node
            self.oldest = new_node
            self.length = 1
            return
        
        if self.latest.timestamp == new_node.timestamp:
            print("Frame with timestamp %.2f already exists"%new_node.timestamp)
            return
        elif self.latest.timestamp > new_node.timestamp:
            raise Exception('Cannot add frame earlier into the past')
        else:
            new_node.prev = self.latest
            self.latest.next = new_node
            self.latest = new_node
            self.length += 1

    def __repr__(self):
        node = self.latest
        nodes = []
        while node is not None:
            nodes.append(repr(node))
            node = node.prev
        nodes.append("None")
        return " || -> || ".join(nodes)


class FeatureManager:
    def __init__(self) -> None:
        self.feature_tracks = {}
        self.frame_track = FrameTrack()
        self.num_tracks = 0
        self.max_num_frames = 10
        self.min_track_length = 3

    def push_feature_points(self, feature_points: list):
        for feature_point in feature_points:
            if feature_point.track_id == -1:
                # print("Adding a new feature track %d"%(self.num_tracks))
                track_id = self.num_tracks
                feature_point.track_id = track_id
                new_track = FeatureTrack(track_id)
                self.feature_tracks[track_id] = new_track
                self.num_tracks += 1
            else:
                track_id = feature_point.track_id
                # print("Add point to an existing feature track %d"%(track_id))
            
            # Update feature track
            self.feature_tracks[track_id].push(feature_point)
            
            # Update frame node
            self.frame_track.latest.add_pt(feature_point)

        self._clean_frame_buffer()

    def _clean_frame_buffer(self):
        if self.frame_track.length <= self.max_num_frames:
            return
        
        while self.frame_track.length > self.max_num_frames:
            # Remove the oldest point feature from tracks
            ids, _, _ = self.frame_track.oldest.get_pts_info()

            # Remove the oldest frame
            self.frame_track.pop()

            # Clean feature tracks
            for k in ids:
                # Track might have been deleted for insufficient length
                if k in self.feature_tracks:
                    self.feature_tracks[k].pop()

                    # If track is too short to estimate the velocity
                    if self.feature_tracks[k].length < self.min_track_length:
                        self.feature_tracks.pop(k)
                
            