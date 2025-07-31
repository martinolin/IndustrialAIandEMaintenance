import os
import pandas as pd
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


#Now perpare the data:
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.utils import resample

import matplotlib.pyplot as plt

from Code3DNN import trainDNN 


from geopy.distance import geodesic

showplot=False

def DistanceFromGPSPoints(coordinate1, coordinate2):
    return geodesic(coordinate1, coordinate2).meters
def DistanceFromGPSPointsRow(row):
    return DistanceFromGPSPoints([row['Latitude'], row['Longitude']],[row['EstimatedLatitude'], row['EstimatedLongitude'] ])

def timeEstimate(s,v):
    if v==0 or s<0.001: #if v=0 standingstill or distance also saying we are standing still we shall just assume connectivity is resumed and data i send with the resciev frequency (the most common found in the estimated)
        return 0
    t=s/(v/3.6) # 3.6 since v is km/h but we want m/s
    return t


def ExtractFeaturesFromVibrationSegments(segments):
            min_1=np.min(segments[:,0])
            min_2=np.min(segments[:,1])
            max_1= np.max(segments[:,0])
            max_2= np.max(segments[:,1])
            variance_1= np.var(segments[:,0])
            variance_2= np.var(segments[:,1])
            crest_factor_1=np.max(np.abs(segments[:,1])/np.mean(np.square(segments[:,0])))
            crest_factor_2=np.max(np.abs(segments[:,1])/np.mean(np.square(segments[:,1])))
            accdiff_1=np.max(np.diff(segments[:,0]))
            accdiff_2=np.max(np.diff(segments[:,1]))


            return min_1,min_2,max_1,max_2,variance_1,variance_2,crest_factor_1,crest_factor_2, accdiff_1, accdiff_2
            
        
  
def GetEventInfo():
    files = {
    "Bridge": "data1/converted_coordinates_Resultat_Bridge.csv",
    "RailJoint": "data1/converted_coordinates_Resultat_RailJoint.csv",
    "Turnout": "data1/converted_coordinates_Turnout.csv"
    }
    data_frames = []
    for category, file in files.items():
        try:
            df = pd.read_csv(file, encoding="utf-8")  # Load CSV with UTF-8 encoding
            df.columns = df.columns.str.strip()  # Strip column names of extra spaces
            if "Latitude" in df.columns and "Longitude" in df.columns:
                df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")  # Convert Latitude to numeric
                df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")  # Convert Longitude to numeric
                df = df[["Latitude", "Longitude"]]  # Select necessary columns
                df["Category"] = category  # Add category column
                data_frames.append(df)
                print(f"Successfully loaded {category} data: {len(df)} rows")
            else:
                print(f"Warning: {category} file does not contain 'Latitude' and 'Longitude' columns.")
                print(f"Available columns: {df.columns.tolist()}")
        except Exception as e:
            print(f"Error loading {category}: {e}")

    # Combine all data
    if data_frames:
        data = pd.concat(data_frames, ignore_index=True)
    else:
        raise ValueError("No valid data found. Check your CSV files.")

    # Debugging: Check if all categories exist
    print("Data counts per category:\n", data["Category"].value_counts())

    # Check if latitude and longitude values are valid
    print("Data summary:\n", data.describe())

    # Check for missing values in the data
    print("Missing values per column:\n", data.isnull().sum())

    # Drop rows with missing Latitude or Longitude values (if any)
    data = data.dropna(subset=["Latitude", "Longitude"])
    return data

def GetGPSTrackAndVibrationsBasedOnVelocity(baseflder, eventInfo,segment_duration_seconds=10):
    files = {
        "latitude": '/GPS.latitude.csv', 
        "longitude": '/GPS.longitude.csv',
        "vibration1":'/CH1_ACCEL1Z1.csv',
        "vibration2":'/CH2_ACCEL1Z2.csv',
        "speed": '/GPS.speed.csv'
    }
    files = {key: baseflder + value for key, value in files.items()}

    # Load each CSV into a DataFrame and add a 'timestamp' using the row index.
    dataframes = {}
    for key, file_path in files.items():
        if file_path:
            df = pd.read_csv(file_path, header=None, names=[key])
            df['timestamp'] = df.index
            dataframes[key] = df

    df_gps = pd.DataFrame()
    # Create GPS DataFrame by merging latitude and longitude.
    if "latitude" in dataframes and "longitude" in dataframes:
        df_gps = pd.merge(dataframes["latitude"], dataframes["longitude"], on="timestamp")
        df_gps = pd.merge(df_gps, dataframes["speed"], on="timestamp")
        # Rename columns for consistency
        df_gps = df_gps.rename(columns={"latitude": "Latitude", "longitude": "Longitude"})
        # Add an index column for use in the interactive plot
        df_gps["PointIndex"] = df_gps.index

    #combine vibrations
    df_vibration_merged = pd.DataFrame()
    if "vibration1" in dataframes and "vibration2" in dataframes:
        df_vibration_merged = pd.merge(
            dataframes["vibration1"],
            dataframes["vibration2"],
            on="timestamp"
            # When the column names differ (here: vibration1 vs vibration2), suffixes are not needed.
        )


    
    distances= [0]  # First row has no previous
    estimatedTime=[]
    maxspeed=np.max(df_gps['speed'])/3.6 #meter/s
    maxindex=df_gps.shape[0]
    estimatedLat=[] #will estimate coordinates in the middle between two points
    estimatedLng=[]
    startLat=[] #will estimate coordinates in the middle between two points
    startLng=[]
    stopLat=[] #will estimate coordinates in the middle between two points
    stopLng=[]
    estimatedaccumulativeTime=[]
    accumulatedTime=0
    for i in range(1, len(df_gps)):
        distance= DistanceFromGPSPoints((df_gps['Latitude'].iloc[i - 1], df_gps['Longitude'].iloc[i-1]),(df_gps['Latitude'].iloc[i], df_gps['Longitude'].iloc[i]))
        
        meanspeed=np.mean([df_gps['speed'].iloc[i - 1], df_gps['speed'].iloc[i]])
        time= timeEstimate(distance, meanspeed)

        #if distance>maxspeed*segment_duration_seconds*1.1 or distance<0.0001: #something is probalby wrong here, we have traveled to far from what should be possible, gps data nolinger synked with vibrations, if the value is very low train stands still and we dont know for how long...)
        if distance>meanspeed*time*1.2+1 or maxspeed*1.1<distance or distance==0: #something is probalby wrong here, we have traveled to far from what should be possible, gps data nolinger synked with vibrations, if the value is very low train stands still and we dont know for how long...)
            print("got a distance of between two GPS points of: " + str(distance) + " index: " + str(i) + ", must have lost connection the GPS and can no longer connect GPS and vibrations")
            maxindex=i
            break
        else:



            
            estimatedLat.append(np.mean([df_gps['Latitude'].iloc[i - 1],df_gps['Latitude'].iloc[i]]))
            estimatedLng.append(np.mean([df_gps['Longitude'].iloc[i - 1],df_gps['Longitude'].iloc[i]]))
            startLat.append(df_gps['Latitude'].iloc[i - 1]) #we get the starting point and then sample 10 secounds after this...
            startLng.append(df_gps['Longitude'].iloc[i - 1])
            stopLat.append(df_gps['Latitude'].iloc[i ]) #we get the starting point and then sample 10 secounds after this...
            stopLng.append(df_gps['Longitude'].iloc[i ])
            distances.append(distance)
            accumulatedTime=accumulatedTime+ time
            estimatedaccumulativeTime.append(accumulatedTime)
        estimatedTime.append(time)
    
    #now lets estimate update freq of GPS
    estimatedTimeRemovedZeros=[x for x in estimatedTime if x != 0]
    if len(estimatedTimeRemovedZeros)==0:
         estimatedUpdateFreq=0
    else:
        estimatedUpdateFreq=np.mean(estimatedTimeRemovedZeros)


    totaldistance=np.sum(distances)
    print("from "+ baseflder+ "we got a total of: " + str(totaldistance) + " meters and "+ str(accumulatedTime) + " seconds of time, based on GPS update freq of every: " + str (estimatedUpdateFreq) + " s")



    #findoutWhich coordinates are usefull and closest to events
    eventInfo['eventID']=eventInfo.index
    GPScoords = pd.DataFrame({'EstimatedLatitude': startLat, 'EstimatedLongitude': startLng, 'index' : range(1, len(startLng)+1), 'accTime':estimatedaccumulativeTime})#,'StartLatitude': startLat, 'StartLongitude': startLng})
    GPScoordsAndEvents= eventInfo.merge(GPScoords, how='cross')
    GPScoordsAndEvents['distance']=   GPScoordsAndEvents.apply( DistanceFromGPSPointsRow,axis=1) #get all combinations of distances possible between events and vibration/GPS
    
    noeventGPScoords= GPScoordsAndEvents.loc[GPScoordsAndEvents.groupby('index')['distance'].idxmin()]
    noeventGPScoords=noeventGPScoords[noeventGPScoords['distance']>1500]
    noeventGPScoords['Category']='noevent'
    closest = GPScoordsAndEvents.loc[GPScoordsAndEvents.groupby('eventID')['distance'].idxmin()]
    

    #even if we found the closest point, we still need to ensure we get the first GPS cord before the event happens. since we have a long sample time we can include the GPS point b4 this one... 



    



    columns=['longitude','latitude','min1','max1','variance1', 'crest_factor1','min2','max2','variance2', 'crest_factor2','distancemissmatch','category', 'eventID', 'accdiff_1', 'accdiff_2']#data columns we want
    myExtractedData=pd.DataFrame(columns=columns) 
     # Data Preprocessing and Segmentation for Vibration Data
    dt_vibration = 0.002  # seconds per sample (e.g. 500 Hz sampling rate)
    
    segment_length = int(segment_duration_seconds / (dt_vibration))
    segments = []
    #num_segments = totalTime // segment_duration_seconds #this is if we can use the estimated time... but since also precence of standing still... we cant..
    num_segments = len(df_vibration_merged) // segment_length
    datapointstogather=np.min([num_segments,len(estimatedLng)]) #nr of datapoints we have that is continues GPS data and vibrationdata..
    noeventGPScoords=noeventGPScoords[noeventGPScoords['index']<datapointstogather]

    noeventGPScoords

    index=0
    for t in range(closest.shape[0]-1):
        #if startLat[i]==closest['EstimatedLatitude'].iloc[t] and startLng[i]==closest['EstimatedLongitude'].iloc[t] and closest['distance'].iloc[t]<50: #the closest datapoint, but still not more than 50 meter off
        if  closest['distance'].iloc[t]<50 and closest['index'].iloc[t]<datapointstogather: #the closest datapoint, but still not more than 50 meter off
            seg = df_vibration_merged.iloc[int((closest['acctime'].iloc[t] -1)/dt_vibration): int((closest['acctime'].iloc[t] -1)/dt_vibration+ segment_length)][["vibration1", "vibration2"]].values
            _min1, _min2, _max1,_max2, _variance1,_variance2, _crest_factor1,_crest_factor2, accdiff_1, accdiff_2= ExtractFeaturesFromVibrationSegments(np.array(seg))
            myExtractedData.loc[index]={'longitude':closest['Longitude'].iloc[t] ,'latitude':closest['Latitude'].iloc[t] ,'min1':_min1,'max1':_max1,'variance1':_variance1, 'crest_factor1':_crest_factor1,'min2':_min2,'max2':_max2,'variance2':_variance2, 'crest_factor2':_crest_factor2, 'distancemissmatch':closest['distance'].iloc[t],'category':closest['Category'].iloc[t],'eventID':closest['eventID'].iloc[t], 'accdiff_1':accdiff_1, 'accdiff_2':accdiff_2}

            index=index+1

    #now we also need some noevent data:
    nrOfNoEventWanted= np.min([ len(myExtractedData), len(noeventGPScoords)])
    for idx, row in noeventGPScoords.sample(n=nrOfNoEventWanted,random_state=21).iterrows(): #may get the same segments several times could even out the sampling by index...but will take too much time
    
        seg = df_vibration_merged.iloc[int((row['acctime'])/dt_vibration):int((row['acctime']+1 )/dt_vibration + segment_length)][["vibration1", "vibration2"]].values

        _min1, _min2, _max1,_max2, _variance1,_variance2, _crest_factor1,_crest_factor2, accdiff_1, accdiff_2= ExtractFeaturesFromVibrationSegments(np.array(seg))
        myExtractedData.loc[index]={'longitude':row['Longitude'] ,'latitude':row['Latitude'] ,'min1':_min1,'max1':_max1,'variance1':_variance1, 'crest_factor1':_crest_factor1,'min2':_min2,'max2':_max2,'variance2':_variance2, 'crest_factor2':_crest_factor2, 'distancemissmatch':closest['distance'].iloc[t],'category':'noevent','eventID':closest['eventID'].iloc[t], 'accdiff_1':accdiff_1, 'accdiff_2':accdiff_2}
        index=index+1


    return myExtractedData



def GetGPSTrackAndVibrations(baseflder, eventInfo,segment_duration_seconds=10,assumedGPSreceiverfreq = 0.5):
    files = {
        "latitude": '/GPS.latitude.csv', 
        "longitude": '/GPS.longitude.csv',
        "vibration1":'/CH1_ACCEL1Z1.csv',
        "vibration2":'/CH2_ACCEL1Z2.csv',
        "speed": '/GPS.speed.csv'
    }
    files = {key: baseflder + value for key, value in files.items()}

    # Load each CSV into a DataFrame and add a 'timestamp' using the row index.
    dataframes = {}
    for key, file_path in files.items():
        if file_path:
            df = pd.read_csv(file_path, header=None, names=[key])
            df['timestamp'] = df.index
            dataframes[key] = df

    df_gps = pd.DataFrame()
    # Create GPS DataFrame by merging latitude and longitude.
    if "latitude" in dataframes and "longitude" in dataframes:
        df_gps = pd.merge(dataframes["latitude"], dataframes["longitude"], on="timestamp")
        df_gps = pd.merge(df_gps, dataframes["speed"], on="timestamp")
        # Rename columns for consistency
        df_gps = df_gps.rename(columns={"latitude": "Latitude", "longitude": "Longitude"})
        # Add an index column for use in the interactive plot
        df_gps["PointIndex"] = df_gps.index

    #combine vibrations
    df_vibration_merged = pd.DataFrame()
    if "vibration1" in dataframes and "vibration2" in dataframes:
        df_vibration_merged = pd.merge(
            dataframes["vibration1"],
            dataframes["vibration2"],
            on="timestamp"
            # When the column names differ (here: vibration1 vs vibration2), suffixes are not needed.
        )


    
    distances= [0]  # First row has no previous
    estimatedTime=[]
    maxspeed=np.max(df_gps['speed'])/3.6 #meter/s
    maxindex=df_gps.shape[0]
    estimatedLat=[] #will estimate coordinates in the middle between two points
    estimatedLng=[]
    startLat=[] #will estimate coordinates in the middle between two points
    startLng=[]
    stopLat=[] #will estimate coordinates in the middle between two points
    stopLng=[]
    estimatedaccumulativeTime=[]
    accumulatedTime=0
    for i in range(1, len(df_gps)):
        distance= DistanceFromGPSPoints((df_gps['Latitude'].iloc[i - 1], df_gps['Longitude'].iloc[i-1]),(df_gps['Latitude'].iloc[i], df_gps['Longitude'].iloc[i]))
        
        meanspeed=np.mean([df_gps['speed'].iloc[i - 1], df_gps['speed'].iloc[i]])
        time= timeEstimate(distance, meanspeed)

        #if distance>maxspeed*segment_duration_seconds*1.1 or distance<0.0001: #something is probalby wrong here, we have traveled to far from what should be possible, gps data nolinger synked with vibrations, if the value is very low train stands still and we dont know for how long...)
        if distance>meanspeed*time*1.2+1 or maxspeed*1.1<distance: #something is probalby wrong here, we have traveled to far from what should be possible, gps data nolinger synked with vibrations, if the value is very low train stands still and we dont know for how long...)
            print("got a distance of between two GPS points of: " + str(distance) + " index: " + str(i) + ", must have lost connection the GPS and can no longer connect GPS and vibrations")
            maxindex=i
            break
        else:



            
            estimatedLat.append(np.mean([df_gps['Latitude'].iloc[i - 1],df_gps['Latitude'].iloc[i]]))
            estimatedLng.append(np.mean([df_gps['Longitude'].iloc[i - 1],df_gps['Longitude'].iloc[i]]))
            startLat.append(df_gps['Latitude'].iloc[i - 1]) #we get the starting point and then sample 10 secounds after this...
            startLng.append(df_gps['Longitude'].iloc[i - 1])
            stopLat.append(df_gps['Latitude'].iloc[i ]) #we get the starting point and then sample 10 secounds after this...
            stopLng.append(df_gps['Longitude'].iloc[i ])
            distances.append(distance)
            accumulatedTime=accumulatedTime+ time
            estimatedaccumulativeTime.append(accumulatedTime)
        estimatedTime.append(time)
    
    #now lets estimate update freq of GPS
    estimatedTimeRemovedZeros=[x for x in estimatedTime if x != 0]
    if len(estimatedTimeRemovedZeros)==0:
         estimatedUpdateFreq=0
    else:
        estimatedUpdateFreq=np.mean(estimatedTimeRemovedZeros)


    totaldistance=np.sum(distances)
    print("from "+ baseflder+ "we got a total of: " + str(totaldistance) + " meters and "+ str(accumulatedTime) + " seconds of time, based on GPS update freq of every: " + str (estimatedUpdateFreq) + " s")



    #findoutWhich coordinates are usefull and closest to events
    eventInfo['eventID']=eventInfo.index
    GPScoords = pd.DataFrame({'EstimatedLatitude': startLat, 'EstimatedLongitude': startLng, 'index' : range(1, len(startLng)+1)})#,'StartLatitude': startLat, 'StartLongitude': startLng})
    GPScoordsAndEvents= eventInfo.merge(GPScoords, how='cross')
    GPScoordsAndEvents['distance']=   GPScoordsAndEvents.apply( DistanceFromGPSPointsRow,axis=1) #get all combinations of distances possible between events and vibration/GPS
    
    noeventGPScoords= GPScoordsAndEvents.loc[GPScoordsAndEvents.groupby('index')['distance'].idxmin()]
    noeventGPScoords=noeventGPScoords[noeventGPScoords['distance']>1500]
    noeventGPScoords['Category']='noevent'
    closest = GPScoordsAndEvents.loc[GPScoordsAndEvents.groupby('eventID')['distance'].idxmin()]
    

    #even if we found the closest point, we still need to ensure we get the first GPS cord before the event happens. since we have a long sample time we can include the GPS point b4 this one... 



    



    columns=['longitude','latitude','min1','max1','variance1', 'crest_factor1','min2','max2','variance2', 'crest_factor2','distancemissmatch','category', 'eventID', 'accdiff_1', 'accdiff_2']#data columns we want
    myExtractedData=pd.DataFrame(columns=columns) 
     # Data Preprocessing and Segmentation for Vibration Data
    dt_vibration = 0.002  # seconds per sample (e.g. 500 Hz sampling rate)
    
    segmentIndexMulitplier=assumedGPSreceiverfreq / (dt_vibration)
    segment_length = int(segment_duration_seconds / (dt_vibration))
    segments = []
    #num_segments = totalTime // segment_duration_seconds #this is if we can use the estimated time... but since also precence of standing still... we cant..
    num_segments = len(df_vibration_merged) // segment_length
    datapointstogather=np.min([num_segments,len(estimatedLng)]) #nr of datapoints we have that is continues GPS data and vibrationdata..
    noeventGPScoords=noeventGPScoords[noeventGPScoords['index']<datapointstogather]

    noeventGPScoords

    index=0
    for t in range(closest.shape[0]-1):
        #if startLat[i]==closest['EstimatedLatitude'].iloc[t] and startLng[i]==closest['EstimatedLongitude'].iloc[t] and closest['distance'].iloc[t]<50: #the closest datapoint, but still not more than 50 meter off
        if  closest['distance'].iloc[t]<50 and closest['index'].iloc[t]<datapointstogather: #the closest datapoint, but still not more than 50 meter off
            seg = df_vibration_merged.iloc[int((closest['index'].iloc[t] -1)* segmentIndexMulitplier): int((closest['index'].iloc[t]-1 ) * segmentIndexMulitplier+ segment_length)][["vibration1", "vibration2"]].values
            _min1, _min2, _max1,_max2, _variance1,_variance2, _crest_factor1,_crest_factor2, accdiff_1, accdiff_2= ExtractFeaturesFromVibrationSegments(np.array(seg))
            myExtractedData.loc[index]={'longitude':closest['Longitude'].iloc[t] ,'latitude':closest['Latitude'].iloc[t] ,'min1':_min1,'max1':_max1,'variance1':_variance1, 'crest_factor1':_crest_factor1,'min2':_min2,'max2':_max2,'variance2':_variance2, 'crest_factor2':_crest_factor2, 'distancemissmatch':closest['distance'].iloc[t],'category':closest['Category'].iloc[t],'eventID':closest['eventID'].iloc[t], 'accdiff_1':accdiff_1, 'accdiff_2':accdiff_2}

            index=index+1

    #now we also need some noevent data:
    nrOfNoEventWanted= np.min([ len(myExtractedData), len(noeventGPScoords)])
    for idx, row in noeventGPScoords.sample(n=nrOfNoEventWanted,random_state=21).iterrows(): #may get the same segments several times could even out the sampling by index...but will take too much time
    
        seg = df_vibration_merged.iloc[int((row['index'])* segmentIndexMulitplier): int((row['index'] ) * segmentIndexMulitplier+ segment_length)][["vibration1", "vibration2"]].values

        _min1, _min2, _max1,_max2, _variance1,_variance2, _crest_factor1,_crest_factor2, accdiff_1, accdiff_2= ExtractFeaturesFromVibrationSegments(np.array(seg))
        myExtractedData.loc[index]={'longitude':row['Longitude'] ,'latitude':row['Latitude'] ,'min1':_min1,'max1':_max1,'variance1':_variance1, 'crest_factor1':_crest_factor1,'min2':_min2,'max2':_max2,'variance2':_variance2, 'crest_factor2':_crest_factor2, 'distancemissmatch':closest['distance'].iloc[t],'category':'noevent','eventID':closest['eventID'].iloc[t], 'accdiff_1':accdiff_1, 'accdiff_2':accdiff_2}
        index=index+1


    return myExtractedData





def plotYdata(y,title):
    # Count label frequencies
    label_counts = y[:].value_counts()

    # Plot histogram-like bar chart
    label_counts.plot(kind='bar')

    # Customize plot
    plt.title('category Frequency Histogram')
    plt.xlabel('categroy')
    plt.ylabel('Count')
    #plt.xticks(rotation=45)
    plt.tight_layout()
    plt.title(title)
    plt.show()




segment_durations=[10,5,3,2]

GPSUpdateFreqs=[-1,1,0.5,0.2, 0.25, 1/3] #-1 means use the velocity data available

myGPSVibrationAccuracyDataFrame=pd.DataFrame(columns=['Assumed GPS freq/s:', 'SVM accuracy:', 'Random Forst accuracy:', 'Deep neural network acc:', 'segment_duration'])


myGPSVibrationDataFrame=pd.DataFrame()

if os.path.exists("eventInfo.csv"):
    eventinfo=pd.read_csv("eventInfo.csv")
else:
    eventinfo=GetEventInfo()
    eventinfo.to_csv("eventInfo.csv")

for segment_duration in segment_durations:
    for GPSupdateFreq in GPSUpdateFreqs:
        if GPSupdateFreq==-1: #then we do based onvelocity instead:
            if os.path.exists("GPSTrackAndVibrationsBasedOnVelocity"+ "segduration"+ str(segment_duration)+ ".csv"):
                myGPSVibrationDataFrame = pd.read_csv("GPSTrackAndVibrationsBasedOnVelocity"+ "segduration"+ str(segment_duration)+ ".csv")
            else:
                myGPSVibrationData=[]
                myGPSVibrationData.append(GetGPSTrackAndVibrations('data2/2024-12-10 12.00',eventinfo,segment_duration))
                myGPSVibrationData.append(GetGPSTrackAndVibrations('data2/2024-12-10 14.00',eventinfo,segment_duration))
                myGPSVibrationData.append(GetGPSTrackAndVibrations('data2/2024-12-10 16.00 (doubletrack)',eventinfo,segment_duration))
                myGPSVibrationData.append(GetGPSTrackAndVibrations('data2/2024-12-11 10.00',eventinfo,segment_duration))
                myGPSVibrationData.append(GetGPSTrackAndVibrations('data2/2024-12-11 18.00',eventinfo,segment_duration))
                myGPSVibrationData.append(GetGPSTrackAndVibrations('data2/2024-12-12 12.00',eventinfo,segment_duration))
                myGPSVibrationDataFrame=pd.concat(myGPSVibrationData, ignore_index=True)

                myGPSVibrationDataFrame.to_csv("GPSTrackAndVibrationsBasedOnVelocity"+ "segduration"+ str(segment_duration)+ ".csv")
        else:
        
            if os.path.exists("GPSandVibrationsCombinedGPSUpdateFreq" +str(GPSupdateFreq) + "segduration"+ str(segment_duration)+ ".csv"):
                myGPSVibrationDataFrame = pd.read_csv("GPSandVibrationsCombinedGPSUpdateFreq" +str(GPSupdateFreq) + "segduration"+ str(segment_duration)+ ".csv")
            else:
                myGPSVibrationData=[]
                myGPSVibrationData.append(GetGPSTrackAndVibrations('data2/2024-12-10 12.00',eventinfo,segment_duration,GPSupdateFreq))
                myGPSVibrationData.append(GetGPSTrackAndVibrations('data2/2024-12-10 14.00',eventinfo,segment_duration,GPSupdateFreq))
                myGPSVibrationData.append(GetGPSTrackAndVibrations('data2/2024-12-10 16.00 (doubletrack)',eventinfo,segment_duration,GPSupdateFreq))
                myGPSVibrationData.append(GetGPSTrackAndVibrations('data2/2024-12-11 10.00',eventinfo,segment_duration,GPSupdateFreq))
                myGPSVibrationData.append(GetGPSTrackAndVibrations('data2/2024-12-11 18.00',eventinfo,segment_duration,GPSupdateFreq))
                myGPSVibrationData.append(GetGPSTrackAndVibrations('data2/2024-12-12 12.00',eventinfo,segment_duration,GPSupdateFreq))

                myGPSVibrationDataFrame=pd.concat(myGPSVibrationData, ignore_index=True)

                myGPSVibrationDataFrame.to_csv("GPSandVibrationsCombinedGPSUpdateFreq" +str(GPSupdateFreq) + "segduration"+ str(segment_duration) + ".csv")

        myGPSVibrationDataFrame.drop_duplicates(subset=['longitude','latitude','min1','max1','variance1', 'crest_factor1','min2','max2','variance2', 'crest_factor2','distancemissmatch','category', 'eventID', 'accdiff_1', 'accdiff_2'], keep='first')

        print("#"*30)
        print("Assumed GPS freq:" +str(GPSupdateFreq))
        y=myGPSVibrationDataFrame['category']

        if showplot:
            print("Gathered data")
            plotYdata(myGPSVibrationDataFrame['category'], "Gathered data")

        

        #balance class data
        df_Turnout = myGPSVibrationDataFrame[myGPSVibrationDataFrame['category'] == 'Turnout']
        df_Brdige = myGPSVibrationDataFrame[myGPSVibrationDataFrame['category'] == 'Bridge']
        df_RailJoint = myGPSVibrationDataFrame[myGPSVibrationDataFrame['category'] == 'RailJoint']
        df_noevent = myGPSVibrationDataFrame[myGPSVibrationDataFrame['category'] == 'noevent']
        nrOfSamples=20
        x =pd.concat([resample(df_Turnout,replace=False,n_samples=nrOfSamples,  random_state=42),resample(df_Brdige,replace=False,n_samples=nrOfSamples,  random_state=42),resample(df_RailJoint,replace=False,n_samples=nrOfSamples,  random_state=42),resample(df_noevent,replace=False,n_samples=nrOfSamples,  random_state=42)]) 

        
        if showplot:
            print("Balanced data")
            plotYdata(x['category'], "Balanced data")

        scaleing=MinMaxScaler()
        y=x['category']
        x=x.drop('category',axis=1)
        x=pd.DataFrame(scaleing.fit_transform(x), columns=x.columns)

        X_train, X_test, Y_train, Y_test = train_test_split(x[['min1','max1','variance1', 'crest_factor1','min2','max2','variance2', 'crest_factor2','accdiff_1', 'accdiff_2']],y,test_size=0.25, random_state=35)
        if showplot:
            print("training data:")
            plotYdata(Y_train, "training data")
            print("testing data:")
            plotYdata(Y_test,"testing data")



        print("train and evaulate DNN")
        accDNN=trainDNN(X_train, X_test, Y_train, Y_test )

        print("")
        print("train SVM:")
        model=SVC()
        model.fit(X_train,Y_train)
        y_pred= model.predict(X_test)
        cm = confusion_matrix(Y_test, y_pred)
        print(cm)
        if showplot:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
            disp.plot(cmap='Blues')
            plt.show()
        accSVM=accuracy_score(Y_test,y_pred) 
        print("acc: " + str(accSVM))


        print("")
        print("train random forest classifier")
        modelrfc_ = RandomForestClassifier(n_estimators=50, random_state=42)
        modelrfc_.fit(X_train, Y_train)
        y_pred= modelrfc_.predict(X_test)
        cm = confusion_matrix(Y_test, y_pred)
        print(cm)
        accRFC=accuracy_score(Y_test,y_pred) 
        print("acc: " + str(accRFC))

        if GPSupdateFreq==-1:
            myGPSVibrationAccuracyDataFrame.loc[len(myGPSVibrationAccuracyDataFrame)]={'Assumed GPS freq/s:': str('GPS-distance/velocity'), 'SVM accuracy:': str(np.round(accSVM,2)), 'Random Forst accuracy:': str(np.round(accRFC,2)), 'Deep neural network acc:':str(np.round(accDNN.item(),2)), 'segment_duration': str(segment_duration)}
        else:
            myGPSVibrationAccuracyDataFrame.loc[len(myGPSVibrationAccuracyDataFrame)]={'Assumed GPS freq/s:': str(np.round(GPSupdateFreq,2)), 'SVM accuracy:': str(np.round(accSVM,2)), 'Random Forst accuracy:': str(np.round(accRFC,2)), 'Deep neural network acc:':str(np.round(accDNN.item(),2)),'segment_duration': str(segment_duration)}


print("results")
print(myGPSVibrationAccuracyDataFrame)

print("best acc per GPS update freq:")
myGPSVibrationAccuracyDataFrame.groupby('Assumed GPS freq/s:')[['SVM accuracy:', 'Random Forst accuracy:', 'Deep neural network acc:']].max()
print("best acc per segment duration")
myGPSVibrationAccuracyDataFrame.groupby('segment_duration')[['SVM accuracy:', 'Random Forst accuracy:', 'Deep neural network acc:']].max()


