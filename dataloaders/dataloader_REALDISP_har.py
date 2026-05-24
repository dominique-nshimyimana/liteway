import pandas as pd
import os

from dataloaders.dataloader_base import BASE_DATA

# ========================================       WEAR_HAR_DATA               =============================
class REALDISP_HAR_DATA(BASE_DATA):
    """
    REALDISP HAR Dataset: http://archive.ics.uci.edu/ml/machine-learning-databases/00305/realistic_sensor_displacement.zip
    URL: https://archive.ics.uci.edu/dataset/305/realdisp+activity+recognition+dataset
    PAPER: https://dl.acm.org/doi/pdf/10.1145/2370216.2370437

    # CODE #
    # https://github.com/sergiosaez6/activity-recognition-realdisp.git #
    # https://github.com/jortboon/sparse-precision-vnn.git #
    # CODE #
    Brief Description of the Dataset:
    ---------------------------------

    The REALDISP (REAListic sensor DISPlacement) dataset has been originally collected to investigate the effects
    of sensor displacement in the activity recognition process in real-world settings.
    It builds on the concept of ideal-placement, self-placement and induced-displacement.
    The ideal and mutual-displacement conditions represent extreme displacement variants and thus could represent
    boundary conditions for recognition algorithms. In contrast, self-placement reflects a users perception of
    how sensors could be attached, e.g., in a sports or lifestyle application. The dataset includes a wide range of
    physical activities (warm up, cool down and fitness exercises), sensor modalities (acceleration, rate of turn,
    magnetic field and quaternions) and participants (17 subjects). Apart from investigating sensor displacement,
    the dataset lend itself for benchmarking activity recognition techniques in ideal conditions.

    Dataset summary:
        Activities: 33
        Sensors: 9 * 13
        Subjects: 17
        Scenarios: 3
        Time: 9 hours in total


    Activities (labels):
        A1: Walking
        A2: Jogging
        A3: Running
        A4: Jump up
        A5: Jump front & back
        A6: Jump sideways
        A7: Jump leg/arms open/closed
        A8: Jump rope
        A9: Trunk twist (arms outstretched)
        A10: Trunk twist (elbows bent)
        A11: Waist bends forward
        A12: Waist rotation
        A13: Waist bends (reach foot with opposite hand)
        A14: Reach heels backwards
        A15: Lateral bend (10_ to the left + 10_ to the right)
        A16: Lateral bend with arm up (10_ to the left + 10_ to the right)
        A17: Repetitive forward stretching
        A18: Upper trunk and lower body opposite twist
        A19: Lateral elevation of arms
        A20: Frontal elevation of arms
        A21: Frontal hand claps
        A22: Frontal crossing of arms
        A23: Shoulders high-amplitude rotation
        A24: Shoulders low-amplitude rotation
        A25: Arms inner rotation
        A26: Knees (alternating) to the breast
        A27: Heels (alternating) to the backside
        A28: Knees bending (crouching)
        A29: Knees (alternating) bending forward
        A30: Rotation on the knees
        A31: Rowing
        A32: Elliptical bike
        A33: Cycling

    Positions (labels) :
        S1: left calf (LC)
        S2: left thigh (LT)
        S3: right calf (RC)
        S4: right thigh (RT)
        S5: back (BACK)
        S6: left lower arm (LLA)
        S7: left upper arm (LUA)
        S8: right lower arm (RLA)
        S9: right upper arm (RUA)

    """

    def __init__(self, args):

        """
        root_path : Root directory of the data set
        difference (bool) : Whether to calculate the first order derivative of the original data
        datanorm_type (str) : Methods of data normalization: "standardization", "minmax" , "per_sample_std", "per_sample_minmax"
        
        spectrogram (bool): Whether to convert raw data into frequency representations
            scales : Depends on the sampling frequency of the data
            wavelet : Methods of wavelet transformation

        """

        # there are total 3 sensors :ACC Gyro Mag
        # amounted in 3 fix + 6 dynamic places "T", "RA", "LA", "RL", "LL"
        # In total 45 Channels

        # Sensors (accX,accY,accZ), 3D gyro (gyrX,gyrY,gyrZ), (magX,magY,magZ) and (Q1,Q2,Q3,Q4).
        # Sensor ordering (as specified in the manual)
        positions = ["RLA", "RUA", "BACK", "LUA", "LLA", "RC", "RT", "LT", "LC"]    # 9 devices

        # Modality ordering (exact column order per sensor)
        sensors = [
            "acc_x", "acc_y", "acc_z",
            "gyr_x", "gyr_y", "gyr_z",
            "mag_x", "mag_y", "mag_z",
            "quat_1", "quat_2", "quat_3", "quat_4"
        ]   # 13 modalities

        all_columns = (
                ['Timestamp_second', 'Timestamp_microsecond'] +
                [f"{sensor}_{pos}" for pos in positions for sensor in sensors] +    # 117=9*13 columns
                ['Activity_label']
        )

        self.used_cols = [2 + m + p*13 for p in range(9) for m in range(9)]+[119]  # except timestamp & quat - columns
        # self.used_cols = list(range(2,120))    # except timestamp - columns
        self.col_names = [all_columns[index] for index in self.used_cols]



        # These two variables represent whether all sensors can be filtered according to position and sensor type
        # pos_filter ------- >  filter according to position
        # sensor_filter ----->  filter according to the sensor type
        self.pos_filter         = ["RLA", "RUA", "BACK", "LUA", "LLA", "RC", "RT", "LT", "LC"]
        self.sensor_filter      = ["acc", "gyr", "mag", "quat"]


        # selected_cols will be updated according to user settings. User have to set -- args.pos_select, args.sensor_select---
        self.selected_cols      = None

        # Filtering channels according to the Position
        if args.pos_select is not None:
            self.selected_cols      = self.Sensor_filter_acoording_to_pos_and_type(args.pos_select, self.pos_filter, self.col_names, "position")

        # Filtering channels according to the Sensor Type
        if args.sensor_select is not None:
            if self.selected_cols is None:
                self.selected_cols  = self.Sensor_filter_acoording_to_pos_and_type(args.sensor_select, self.sensor_filter, self.col_names, "Sensor Type")
            else:
                self.selected_cols  = self.Sensor_filter_acoording_to_pos_and_type(args.sensor_select, self.sensor_filter, self.selected_cols, "Sensor Type")

        # self.selected_cols = [...]
        # self.selected_cols = [...]
        # print(f"{self.selected_cols} --- self.col_names: {self.col_names}");exit()

        self.label_map = [
            (0, "null"),    # Not listed there
            (1, "Walking"),
            (2, "Jogging"),
            (3, "Running"),
            (4, "Jump up"),
            (5, "Jump front & back"),
            (6, "Jump sideways"),
            (7, "Jump leg/arms open/closed"),
            (8, "Jump rope"),
            (9, "Trunk twist (arms outstretched)"),
            (10, "Trunk twist (elbows bent)"),
            (11, "Waist bends forward"),
            (12, "Waist rotation"),
            (13, "Waist bends (reach foot with opposite hand)"),
            (14, "Reach heels backwards"),
            (15, "Lateral bend (10° left + 10° right)"),
            (16, "Lateral bend with arm up (10° left + 10° right)"),
            (17, "Repetitive forward stretching"),
            (18, "Upper trunk and lower body opposite twist"),
            (19, "Lateral elevation of arms"),
            (20, "Frontal elevation of arms"),
            (21, "Frontal hand claps"),
            (22, "Frontal crossing of arms"),
            (23, "Shoulders high-amplitude rotation"),
            (24, "Shoulders low-amplitude rotation"),
            (25, "Arms inner rotation"),
            (26, "Knees (alternating) to the breast"),
            (27, "Heels (alternating) to the backside"),
            (28, "Knees bending (crouching)"),
            (29, "Knees (alternating) bending forward"),
            (30, "Rotation on the knees"),
            (31, "Rowing"),
            (32, "Elliptical bike"),
            (33, "Cycling"),
        ]

        self.drop_activities = [0]   # labels_to_keep = [10, 11, 29, 31, 9, 32, 33, 3, 2, 1]

        # The keys for each set will be updated in the readtheload function
        self.train_keys   = [1,3,4,5,7,8,9,10,11,13,14,15,17]
        self.vali_keys    = []
        self.test_keys    = [2,6,12,16]

        self.exp_mode     = args.exp_mode
        if self.exp_mode == "LOCV":
            self.split_tag = "sub"
        else:
            self.split_tag = "sub_id"


        self.LOCV_keys = [[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15],[16,17,]]
        self.all_keys = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
        self.sub_ids_of_each_sub = {}

        self.exp_mode     = args.exp_mode
        self.split_tag = "sub"

        self.file_encoding = {}  # no use 
        

        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map)}
        self.all_labels = list(range(len(self.label_map)))

        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]

        super(REALDISP_HAR_DATA, self).__init__(args)

    def load_all_the_data(self, root_path):

        print(" ----------------------- load all the data -------------------")

        df_dict = {}

        for sub_file in os.listdir(root_path):
            if not sub_file.endswith(".log"):     # ".log" for all scenario or "_ideal.log" for ideal
                continue

            subject_part, sess_str = sub_file.replace(".log", "").split("_")
            sub_str = subject_part.replace("subject", "")

            # sess_str is the placement: e.g. self, ideal, mutural4, mutual5, ...
            sub_id = sub_str + sess_str
            sub = int(sub_str)

            sub_data = pd.read_csv(os.path.join(root_path, sub_file), delim_whitespace=True, header=None)
            # print(sub_data.shape, sub_data.columns, '=========>>>>>')
            sub_data = sub_data.iloc[:,self.used_cols]
            sub_data.columns = self.col_names
            # print(sub_data.shape, sub_data.columns, '=========>>>>>')

            sub_data["sub_id"] = sub_id
            sub_data["sub"] = sub

            if sub not in self.sub_ids_of_each_sub.keys():
                self.sub_ids_of_each_sub[sub] = []
            self.sub_ids_of_each_sub[sub].append(sub_id)

            # Handle NAN
            if sub_id in ['15mutual7', ]:
                nan_columns = self.col_names[:-3]
                sub_data[nan_columns] = (
                    sub_data.groupby('sub', group_keys=False)[nan_columns]
                    .apply(lambda g: g.interpolate(method='linear', limit_direction='both'))
                )

            df_dict[sub_id] = sub_data

            # Check for NAN
            if sub_data.isna().any().any():
                # print(f"NAN Checkin ** Sub+Sess:{sub_id} -> len{len(sub_data)} ==> {sub_data.isna().sum()}")
                print(f"NAN Checkin ** Sub+Sess:{sub_id} -> len: {sub_data.shape} ==> {sub_data.isna().sum().sum()}")
                for col, na_count in zip(sub_data.columns, sub_data.isna().sum()):
                    if na_count > 0:
                        print(f"{col}: {na_count}")
                print(sub_data.info(verbose=True))

        df_all = pd.concat(df_dict)
        df_all = df_all.set_index('sub_id')

        # Label Mapping: already in 0-N order
        # print(f"Label counts:\n{df_all['Activity_label'].value_counts()}")
        df_all["activity_id"] = df_all["Activity_label"]
        # df_all["activity_id"] = df_all["Activity_label"].map({label: idx for idx, label in self.label_map})
        df_all["activity_id"] = df_all["activity_id"].map(self.labelToId)

        # reorder the columns as sensor1, sensor2... sensorn, sub, activity_id
        if self.selected_cols:
            df_all = df_all[self.selected_cols+["sub"]+["activity_id"]]
        else:
            df_all = df_all[self.col_names[:-1]+["sub"]+["activity_id"]]


        # print(df_all.attrs)
        # df_all.attrs.update({'labelmap': self.label_map, 'label2id': self.labelToId})
        # print(df_all.attrs)
        # df_all.to_csv(f"datasets/csv/{self.data_name}.csv");print(df_all.shape);exit()

        data_y = df_all.iloc[:,-1]
        data_x = df_all.iloc[:,:-1]

        data_x = data_x.reset_index()
        # sub_id, sensor1, sensor2... sensorn, sub,

        print('Dataset summary:', data_y.shape, data_x.shape, data_x.columns, "\nSubjects-Sessions:", self.sub_ids_of_each_sub, )

        return data_x, data_y


