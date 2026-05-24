import pandas as pd
import os

from dataloaders.dataloader_base import BASE_DATA

# ========================================       WEAR_HAR_DATA               =============================
class WEAR_HAR_DATA(BASE_DATA):
    """
    WEAR:An Outdoor Sports Dataset for Wearable and Egocentric Activity Recognition
    Github: https://mariusbock.github.io/wear/
    URL(50HZ IMU download): https://uni-siegen.sciebo.de/public.php/dav/files/enHPo7HwP8RccAe/raw/inertial/50hz/?accept=zip

    2nd WEAR Dataset Challenge @HASCA 2025
    2nd WEAR Dataset Challenge hosted at the 13th International Workshop on Human Activity Sensing Corpu

    Brief Description of the Dataset:
    ---------------------------------

    3D-accelerometer data (50Hz±8g) was collected using four open-source Bangle.js smartwatches running a custom,
    open-source firmware. The watches and were placed in a fixed orientation on
    the left and right wrists and ankles of each participant.

    Data from 22 participants performing a total of 18 different workout activities was collected with
    synchronized inertial (acceleration) and camera (egocentric video) data recorded at 11 different outside locations.

        Activities (labels):
        Positions (labels) : left wrist, right wrist, left ankle, right ankle
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

        all_columns = ['sbj_id',
                       'right_arm_acc_x', 'right_arm_acc_y', 'right_arm_acc_z',
                       'right_leg_acc_x', 'right_leg_acc_y', 'right_leg_acc_z',
                       'left_leg_acc_x', 'left_leg_acc_y', 'left_leg_acc_z',
                       'left_arm_acc_x', 'left_arm_acc_y', 'left_arm_acc_z',
                       'label'] # renamed

        all_columns = ['sbj_id',
                       'acc_x_right_arm', 'acc_y_right_arm', 'acc_z_right_arm',
                       'acc_x_right_leg', 'acc_y_right_leg', 'acc_z_right_leg',
                       'acc_x_left_leg', 'acc_y_left_leg', 'acc_z_left_leg',
                       'acc_x_left_arm', 'acc_y_left_arm', 'acc_z_left_arm',
                       'label']
        self.used_cols = [0,
                          1,  2,  3,
                          4,  5,  6,
                          7,  8,  9,
                          10, 11, 12,
                          13,]   # *Activity*, Position, syncingTime
        self.col_names = [all_columns[index] for index in self.used_cols]



        # These two variables represent whether all sensors can be filtered according to position and sensor type
        # pos_filter ------- >  filter according to position
        # sensor_filter ----->  filter according to the sensor type
        self.pos_filter         = ["right_arm", "left_arm", "right_leg", "left_leg"]
        self.sensor_filter      = ["acc", ]


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

        # self.selected_cols = [....]
        # self.selected_cols = [....]
        # print(f"{self.selected_cols} --- self.col_names: {self.col_names}");exit()

        self.label_map = [
            (0, "null"),
            (1, "jogging"),
            (2, "jogging (rotating arms)"),
            (3, "jogging (skipping)"),
            (4, "jogging (sidesteps)"),
            (5, "jogging (butt-kicks)"),
            (6, "stretching (triceps)"),
            (7, "stretching (lunging)"),
            (8, "stretching (shoulders)"),
            (9, "stretching (hamstrings)"),
            (10, "stretching (lumbar rotation)"),
            (11, "push-ups"),
            (12, "push-ups (complex)"),
            (13, "sit-ups"),
            (14, "sit-ups (complex)"),
            (15, "burpees"),
            (16, "lunges"),
            (17, "lunges (complex)"),
            (18, "bench-dips"),
        ]

        self.drop_activities = []

        # The keys for each set will be updated in the readtheload function
        self.train_keys   = [1,2,3,4,5,7,8,9,10,11,13,14,15,16,17,19,20,21,22]
        self.vali_keys    = []
        self.test_keys    = [0,6,12,18,20,23]

        self.exp_mode     = args.exp_mode
        if self.exp_mode == "LOCV":
            self.split_tag = "sub"
        else:
            self.split_tag = "sub_id"


        self.LOCV_keys = [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15],[16,17,18,19],[20,21,22,23]]
        self.all_keys = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
        self.sub_ids_of_each_sub = {}

        self.exp_mode     = args.exp_mode
        self.split_tag = "sub"

        self.file_encoding = {}  # no use 
        

        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map)}
        self.all_labels = list(range(len(self.label_map)))

        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]

        super(WEAR_HAR_DATA, self).__init__(args)

    def load_all_the_data(self, root_path):

        print(" ----------------------- load all the data -------------------")

        df_dict = {}

        for sub_file in os.listdir(root_path):

            sub_str = sub_file.split("_")[-1].split(".")[0]
            sess_str = "" if len(sub_file.split("_")) == 2 else sub_file.split("_")[-2]

            sub_id = sub_str + sess_str
            sub = int(sub_str)

            sub_data = pd.read_csv(os.path.join(root_path, sub_file))
            sub_data = sub_data.iloc[:,self.used_cols]
            sub_data.columns = self.col_names

            # Fill "null" for empty class in column label
            sub_data["label"].fillna("null", inplace=True)

            sub_data["sub_id"] = sub_id
            # sub_data["sub"] = sub # rename, since it is there
            sub_data = sub_data.rename(columns={"sbj_id": "sub"})

            if sub not in self.sub_ids_of_each_sub.keys():
                self.sub_ids_of_each_sub[sub] = []
            self.sub_ids_of_each_sub[sub].append(sub_id)

            if sub_id in ['10', ]:
                nan_columns = ["acc_x_left_arm", "acc_y_left_arm", "acc_z_left_arm"]
                sub_data[nan_columns] = (
                    sub_data.groupby('sub', group_keys=False)[nan_columns]
                    .apply(lambda g: g.interpolate(method='linear', limit_direction='both'))
                )

            df_dict[sub_id] = sub_data

            if sub_data.isna().any().any():
                print(f"NAN Checkin ** Sub+Sess:{sub_id} -> len{len(sub_data)} ==> {sub_data.isna().sum()}")

        df_all = pd.concat(df_dict)
        df_all = df_all.set_index('sub_id')


        df_all["activity_id"] = df_all["label"].map({label: idx for idx, label in self.label_map})
        df_all["activity_id"] = df_all["activity_id"].map(self.labelToId)

        # reorder the columns as sensor1, sensor2... sensorn, sub, activity_id
        if self.selected_cols:
            df_all = df_all[self.selected_cols+["sub"]+["activity_id"]]
        else:
            df_all = df_all[self.col_names[1:-1]+["sub"]+["activity_id"]]


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


