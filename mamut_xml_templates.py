# Template pre-spots (root and features).

begin_template = '''<?xml version="1.0" encoding="UTF-8"?>
<TrackMate version="7.11.1">
  <Model spatialunits="px" timeunits="frame">
    <FeatureDeclarations>
      <SpotFeatures>
        <Feature feature="QUALITY" name="Quality" shortname="Quality" dimension="QUALITY" isint="false" />
        <Feature feature="POSITION_X" name="X" shortname="X" dimension="POSITION" isint="false" />
        <Feature feature="POSITION_Y" name="Y" shortname="Y" dimension="POSITION" isint="false" />
        <Feature feature="POSITION_Z" name="Z" shortname="Z" dimension="POSITION" isint="false" />
        <Feature feature="POSITION_T" name="T" shortname="T" dimension="TIME" isint="false" />
        <Feature feature="FRAME" name="Frame" shortname="Frame" dimension="NONE" isint="true" />
        <Feature feature="RADIUS" name="Radius" shortname="R" dimension="LENGTH" isint="false" />
        <Feature feature="VISIBILITY" name="Visibility" shortname="Visibility" dimension="NONE" isint="true" />
        <Feature feature="CELL_DIVISION_TIME" name="Cell division time" shortname="Cell div. time" dimension="TIME" isint="false" />
        <Feature feature="SOURCE_ID" name="Source ID" shortname="Source" dimension="NONE" isint="true" />
        <Feature feature="MANUAL_SPOT_COLOR" name="Manual spot color" shortname="Spot color" dimension="NONE" isint="true" />
        <Feature feature="MEAN_INTENSITY_CH1" name="Mean intensity ch1" shortname="Mean ch1" dimension="INTENSITY" isint="false" />
        <Feature feature="MEDIAN_INTENSITY_CH1" name="Median intensity ch1" shortname="Median ch1" dimension="INTENSITY" isint="false" />
        <Feature feature="MIN_INTENSITY_CH1" name="Min intensity ch1" shortname="Min ch1" dimension="INTENSITY" isint="false" />
        <Feature feature="MAX_INTENSITY_CH1" name="Max intensity ch1" shortname="Max ch1" dimension="INTENSITY" isint="false" />
        <Feature feature="TOTAL_INTENSITY_CH1" name="Sum intensity ch1" shortname="Sum ch1" dimension="INTENSITY" isint="false" />
        <Feature feature="STD_INTENSITY_CH1" name="Std intensity ch1" shortname="Std ch1" dimension="INTENSITY" isint="false" />
        <Feature feature="MEAN_INTENSITY_CH2" name="Mean intensity ch2" shortname="Mean ch2" dimension="INTENSITY" isint="false" />
        <Feature feature="MEDIAN_INTENSITY_CH2" name="Median intensity ch2" shortname="Median ch2" dimension="INTENSITY" isint="false" />
        <Feature feature="MIN_INTENSITY_CH2" name="Min intensity ch2" shortname="Min ch2" dimension="INTENSITY" isint="false" />
        <Feature feature="MAX_INTENSITY_CH2" name="Max intensity ch2" shortname="Max ch2" dimension="INTENSITY" isint="false" />
        <Feature feature="TOTAL_INTENSITY_CH2" name="Sum intensity ch2" shortname="Sum ch2" dimension="INTENSITY" isint="false" />
        <Feature feature="STD_INTENSITY_CH2" name="Std intensity ch2" shortname="Std ch2" dimension="INTENSITY" isint="false" />
        <Feature feature="CONTRAST_CH1" name="Contrast ch1" shortname="Ctrst ch1" dimension="NONE" isint="false" />
        <Feature feature="SNR_CH1" name="Signal/Noise ratio ch1" shortname="SNR ch1" dimension="NONE" isint="false" />
        <Feature feature="CONTRAST_CH2" name="Contrast ch2" shortname="Ctrst ch2" dimension="NONE" isint="false" />
        <Feature feature="SNR_CH2" name="Signal/Noise ratio ch2" shortname="SNR ch2" dimension="NONE" isint="false" />
        <Feature feature="TISSUE_NAME" name="Tissue name" shortname="Tis name" dimension="NONE" isint="false" />
        <Feature feature="TISSUE_TYPE" name="Tissue id" shortname="Tis id" dimension="NONE" isint="true" />
      </SpotFeatures>
      <EdgeFeatures>
        <Feature feature="SPOT_SOURCE_ID" name="Source spot ID" shortname="Source ID" dimension="NONE" isint="true" />
        <Feature feature="SPOT_TARGET_ID" name="Target spot ID" shortname="Target ID" dimension="NONE" isint="true" />
        <Feature feature="LINK_COST" name="Edge cost" shortname="Cost" dimension="COST" isint="false" />
        <Feature feature="DIRECTIONAL_CHANGE_RATE" name="Directional change rate" shortname="γ rate" dimension="ANGLE_RATE" isint="false" />
        <Feature feature="SPEED" name="Speed" shortname="Speed" dimension="VELOCITY" isint="false" />
        <Feature feature="DISPLACEMENT" name="Displacement" shortname="Disp." dimension="LENGTH" isint="false" />
        <Feature feature="EDGE_TIME" name="Edge time" shortname="Edge T" dimension="TIME" isint="false" />
        <Feature feature="EDGE_X_LOCATION" name="Edge X" shortname="Edge X" dimension="POSITION" isint="false" />
        <Feature feature="EDGE_Y_LOCATION" name="Edge Y" shortname="Edge Y" dimension="POSITION" isint="false" />
        <Feature feature="EDGE_Z_LOCATION" name="Edge Z" shortname="Edge Z" dimension="POSITION" isint="false" />
        <Feature feature="MANUAL_EDGE_COLOR" name="Manual edge color" shortname="Edge color" dimension="NONE" isint="true" />
        <Feature feature="TISSUE_NAME" name="Tissue name" shortname="Tis name" dimension="NONE" isint="false" />
        <Feature feature="TISSUE_TYPE" name="Tissue id" shortname="Tis id" dimension="NONE" isint="true" />
      </EdgeFeatures>
      <TrackFeatures>
        <Feature feature="TRACK_INDEX" name="Track index" shortname="Index" dimension="NONE" isint="true" />
        <Feature feature="TRACK_ID" name="Track ID" shortname="ID" dimension="NONE" isint="true" />
        <Feature feature="DIVISION_TIME_MEAN" name="Mean cell division time" shortname="Mean div. time" dimension="TIME" isint="false" />
        <Feature feature="DIVISION_TIME_STD" name="Std cell division time" shortname="Std div. time" dimension="TIME" isint="false" />
        <Feature feature="NUMBER_SPOTS" name="Number of spots in track" shortname="N spots" dimension="NONE" isint="true" />
        <Feature feature="NUMBER_GAPS" name="Number of gaps" shortname="N gaps" dimension="NONE" isint="true" />
        <Feature feature="NUMBER_SPLITS" name="Number of split events" shortname="N splits" dimension="NONE" isint="true" />
        <Feature feature="NUMBER_MERGES" name="Number of merge events" shortname="N merges" dimension="NONE" isint="true" />
        <Feature feature="NUMBER_COMPLEX" name="Number of complex points" shortname="N complex" dimension="NONE" isint="true" />
        <Feature feature="LONGEST_GAP" name="Longest gap" shortname="Lgst gap" dimension="NONE" isint="true" />
        <Feature feature="TRACK_DURATION" name="Track duration" shortname="Duration" dimension="TIME" isint="false" />
        <Feature feature="TRACK_START" name="Track start" shortname="Track start" dimension="TIME" isint="false" />
        <Feature feature="TRACK_STOP" name="Track stop" shortname="Track stop" dimension="TIME" isint="false" />
        <Feature feature="TRACK_DISPLACEMENT" name="Track displacement" shortname="Track disp." dimension="LENGTH" isint="false" />
        <Feature feature="TRACK_X_LOCATION" name="Track mean X" shortname="Track X" dimension="POSITION" isint="false" />
        <Feature feature="TRACK_Y_LOCATION" name="Track mean Y" shortname="Track Y" dimension="POSITION" isint="false" />
        <Feature feature="TRACK_Z_LOCATION" name="Track mean Z" shortname="Track Z" dimension="POSITION" isint="false" />
        <Feature feature="TRACK_MEAN_SPEED" name="Track mean speed" shortname="Mean sp." dimension="VELOCITY" isint="false" />
        <Feature feature="TRACK_MAX_SPEED" name="Track max speed" shortname="Max speed" dimension="VELOCITY" isint="false" />
        <Feature feature="TRACK_MIN_SPEED" name="Track min speed" shortname="Min speed" dimension="VELOCITY" isint="false" />
        <Feature feature="TRACK_MEDIAN_SPEED" name="Track median speed" shortname="Med. speed" dimension="VELOCITY" isint="false" />
        <Feature feature="TRACK_STD_SPEED" name="Track std speed" shortname="Std speed" dimension="VELOCITY" isint="false" />
        <Feature feature="TRACK_MEAN_QUALITY" name="Track mean quality" shortname="Mean Q" dimension="QUALITY" isint="false" />
        <Feature feature="TOTAL_DISTANCE_TRAVELED" name="Total distance traveled" shortname="Total dist." dimension="LENGTH" isint="false" />
        <Feature feature="MAX_DISTANCE_TRAVELED" name="Max distance traveled" shortname="Max dist." dimension="LENGTH" isint="false" />
        <Feature feature="CONFINEMENT_RATIO" name="Confinement ratio" shortname="Cfn. ratio" dimension="NONE" isint="false" />
        <Feature feature="MEAN_STRAIGHT_LINE_SPEED" name="Mean straight line speed" shortname="Mn. v. line" dimension="VELOCITY" isint="false" />
        <Feature feature="LINEARITY_OF_FORWARD_PROGRESSION" name="Linearity of forward progression" shortname="Fwd. progr." dimension="NONE" isint="false" />
        <Feature feature="MEAN_DIRECTIONAL_CHANGE_RATE" name="Mean directional change rate" shortname="Mn. γ rate" dimension="ANGLE_RATE" isint="false" />
      </TrackFeatures>
    </FeatureDeclarations>\n'''


# Templates for spots.
allspots_template =     '    <AllSpots nspots="{nspots}">\n'
inframe_template =      '     <SpotsInFrame frame="{frame}">\n'
spot_template =         '        <Spot ID="{id}" name="{name} SPOT_{id}" VISIBILITY="1" RADIUS="10.0" QUALITY="-1.0" SOURCE_ID="0" POSITION_T="{frame}.0" POSITION_X="{x}" POSITION_Y="{y}" FRAME="{frame}" POSITION_Z="{z}" TISSUE_TYPE="{t_id}" TISSUE_NAME="{t_name}" MANUAL_SPOT_COLOR="{t_color}" />\n'
inframe_end_template =  '     </SpotsInFrame>\n'
allspots_end_template = '    </AllSpots>\n'
inframe_empty_template = '     <SpotsInFrame frame="{frame}" />\n'

# Templates for tracks and edges.
alltracks_template =        '    <AllTracks>\n'
track_template =            '      <Track name="Track_{id}" TRACK_INDEX="{id}" TRACK_ID="{id}" TRACK_DURATION="{duration}.0" TRACK_START="{start}" TRACK_STOP="{stop}.0" TRACK_DISPLACEMENT="{displacement}" NUMBER_SPOTS="{nspots}" NUMBER_GAPS="0" LONGEST_GAP="0" NUMBER_SPLITS="0" NUMBER_MERGES="0" NUMBER_COMPLEX="0" DIVISION_TIME_MEAN="NaN" DIVISION_TIME_STD="NaN">\n'
edge_template =             '        <Edge SPOT_SOURCE_ID="{source_id}" SPOT_TARGET_ID="{target_id}" LINK_COST="-1.0" VELOCITY="{velocity}" DISPLACEMENT="{displacement}" TISSUE_TYPE="{t_id}" TIME="{time}" TISSUE_NAME="{t_name}" MANUAL_EDGE_COLOR="{t_color}" />\n'
# edge_template =             '        <Edge SPOT_SOURCE_ID="{source_id}" SPOT_TARGET_ID="{target_id}" LINK_COST="-1.0" VELOCITY="{velocity}" DISPLACEMENT="{displacement}" />\n'
track_end_template =        '      </Track>\n'
alltracks_end_template =    '    </AllTracks>\n'

# Templates for filtered tracks.
filteredtracks_start_template = '    <FilteredTracks>\n'
filteredtracks_template = '      <TrackID TRACK_ID="{t_id}" />\n'
filteredtracks_end_template = '    </FilteredTracks>\n'

      # <TrackID TRACK_ID="1" />\n      <TrackID TRACK_ID="2" />\n    </FilteredTracks>'

# Template for ending the XML file.
im_data_template = '<ImageData filename="{filename}" folder="{folder}" width="0" height="0" nslices="0" nframes="0" pixelwidth="1.0" pixelheight="1.0" voxeldepth="1.0" timeinterval="1.0" />'
end_template = '''  </Model>
  <Settings>
    {image_data}
    <InitialSpotFilter feature="QUALITY" value="0.0" isabove="true" />
    <SpotFilterCollection />
    <TrackFilterCollection />
    <AnalyzerCollection>
      <SpotAnalyzers>
        <Analyzer key="Cell division time on spots" />
        <Analyzer key="Spot Source ID" />
        <Analyzer key="Manual spot color" />
        <Analyzer key="Spot intensity" />
        <Analyzer key="Spot contrast and SNR" />
      </SpotAnalyzers>
      <EdgeAnalyzers>
        <Analyzer key="Directional change" />
        <Analyzer key="Edge speed" />
        <Analyzer key="Edge target" />
        <Analyzer key="Edge location" />
        <Analyzer key="Manual edge color" />
      </EdgeAnalyzers>
      <TrackAnalyzers>
        <Analyzer key="CELL_DIVISION_TIME_ANALYZER" />
        <Analyzer key="Branching analyzer" />
        <Analyzer key="Track duration" />
        <Analyzer key="Track index" />
        <Analyzer key="Track location" />
        <Analyzer key="Track speed" />
        <Analyzer key="Track quality" />
        <Analyzer key="Track motility analysis" />
      </TrackAnalyzers>
    </AnalyzerCollection>
  </Settings>
  <GUIState>
    <View key="MaMuT Viewer" x="926" y="270" width="790" height="600" />
    <SetupAssignments>
      <ConverterSetups>
        <ConverterSetup>
          <id>0</id>
          <min>10.0</min>
          <max>70.0</max>
          <color>-1</color>
          <groupId>0</groupId>
        </ConverterSetup>
        <ConverterSetup>
          <id>1</id>
          <min>10.0</min>
          <max>90.0</max>
          <color>-1</color>
          <groupId>1</groupId>
        </ConverterSetup>
      </ConverterSetups>
      <MinMaxGroups>
        <MinMaxGroup>
          <id>0</id>
          <fullRangeMin>-2.147483648E9</fullRangeMin>
          <fullRangeMax>2.147483647E9</fullRangeMax>
          <rangeMin>0.0</rangeMin>
          <rangeMax>65535.0</rangeMax>
          <currentMin>10.0</currentMin>
          <currentMax>70.0</currentMax>
        </MinMaxGroup>
        <MinMaxGroup>
          <id>1</id>
          <fullRangeMin>-2.147483648E9</fullRangeMin>
          <fullRangeMax>2.147483647E9</fullRangeMax>
          <rangeMin>0.0</rangeMin>
          <rangeMax>65535.0</rangeMax>
          <currentMin>10.0</currentMin>
          <currentMax>90.0</currentMax>
        </MinMaxGroup>
      </MinMaxGroups>
    </SetupAssignments>
    <Bookmarks />
  </GUIState>
  <DisplaySettings>{{
  "name": "User-default",
  "spotUniformColor": "204, 51, 204, 255",
  "spotColorByType": "SPOTS",
  "spotColorByFeature": "MANUAL_SPOT_COLOR",
  "spotDisplayRadius": 1.0,
  "spotDisplayedAsRoi": true,
  "spotMin": 0.0,
  "spotMax": 10.0,
  "spotShowName": false,
  "trackMin": 0.0,
  "trackMax": 10.0,
  "trackColorByType": "EDGES",
  "trackColorByFeature": "MANUAL_EDGE_COLOR",
  "trackUniformColor": "204, 204, 51, 255",
  "undefinedValueColor": "0, 0, 0, 255",
  "missingValueColor": "89, 89, 89, 255",
  "highlightColor": "51, 230, 51, 255",
  "trackDisplayMode": "FULL",
  "colormap": "Jet",
  "limitZDrawingDepth": false,
  "drawingZDepth": 10.0,
  "fadeTracks": true,
  "fadeTrackRange": 30,
  "useAntialiasing": true,
  "spotVisible": true,
  "trackVisible": true,
  "font": {{
    "name": "Arial",
    "style": 1,
    "size": 12,
    "pointSize": 12.0,
    "fontSerializedDataVersion": 1
  }},
  "lineThickness": 1.0,
  "selectionLineThickness": 4.0,
  "trackschemeBackgroundColor1": "128, 128, 128, 255",
  "trackschemeBackgroundColor2": "192, 192, 192, 255",
  "trackschemeForegroundColor": "0, 0, 0, 255",
  "trackschemeDecorationColor": "0, 0, 0, 255",
  "trackschemeFillBox": false,
  "spotFilled": false,
  "spotTransparencyAlpha": 1.0
}}</DisplaySettings>
</TrackMate>
'''
