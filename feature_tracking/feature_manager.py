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
    def __init__(self, timestamp: float):
        super().__init__()
        self.timestamp = timestamp
        self.pts = {}

    def add_pt(self, point_feature: FeaturePoint):
        if point_feature.timestamp != self.timestamp:
            raise Exception("Mismatch between FeatureFrame time and FeaturePoint")
        
        self.pts[point_feature.track_id] = point_feature

    def get_pts_info(self):
        track_ids, pos, pix = [], [], []
        for i, pt in self.pts.items():
            track_ids.append(i)
            pos.append(pt.pos)
            pix.append(pt.pix)
        return track_ids, pos, pix

    def __repr__(self):
        return "%d pts"%(len(self.pts))

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
        return " -> ".join(nodes)


class FeatureManager:
    def __init__(self) -> None:
        self.feature_tracks = {}
        self.frame_track = FrameTrack()
        ############# To Do ##################
        ## Num Tracks
        self.num_tracks = 0
        self.max_num_frames = 40

    def add_feature_points(self, feature_points: list):
        print("****** FeatureManager, adding new frame with",
              " %d feature points"%(len(feature_points)))
        timestamp = feature_points[0].timestamp
        self.frame_track.push(FrameNode(timestamp))
        
        for feature_pt in feature_points:
            k = feature_pt.track_id
            if k in self.feature_tracks:
                self.feature_tracks[k].push(feature_pt)
                print("Add a new point to track %d"%k)
            else:
                new_track = FeatureTrack(k)
                new_track.push(feature_pt)
                self.feature_tracks[k] = new_track
                print("Init a new track %d"%k)

        self._clean_frame_buffer()

    def _clean_frame_buffer(self):
        if self.frame_track.length <= self.max_num_frames:
            return
        
        while self.frame_track.length > self.max_num_frames:
            # Remove the oldest point feature from tracks
            ids, _, _ = self.frame_track.oldest.get_pts_info()
            for k in ids:
                self.feature_tracks[k].pop()
                # If track is too short to estimate the velocity
                if self.feature_tracks[k].length < 2:
                    self.feature_tracks.pop(k)
            
            # Remove the oldest frame
            self.frame_track.pop()