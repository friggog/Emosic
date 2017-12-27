import sys

import coremltools


def convert(t, path):
    if t == 'C':
        coreml_model = coremltools.converters.keras.convert(path + '/AFF_NET_C_O.h5',
                                                            input_names=['image'],
                                                            image_input_names=['image'],
                                                            output_names=['emotion_p'],
                                                            image_scale=1 / 255.0,
                                                            class_labels=['0', '1', '2', '3', '4', '5', '6', '7'],
                                                            predicted_feature_name='emotion')
        coreml_model.short_description = 'Predicts the emotion present in an image of a human face.'
        coreml_model.input_description['image'] = '96x96 image of human face'
        coreml_model.output_description['emotion'] = 'Predicted emotion - 1 of 8 basic emotions'
    else:
        coreml_model = coremltools.converters.keras.convert(path + '/AFF_NET_R_O.h5',
                                                            input_names=['image'],
                                                            image_input_names=['image'],
                                                            image_scale=1 / 255.0,
                                                            output_names=['valence/arousal'],
                                                            predicted_feature_name='emotion')
        coreml_model.short_description = 'Predicts the valence/arousal present in an image of a human face.'
        coreml_model.input_description['image'] = '96x96 image of human face'
        coreml_model.output_description['valence/arousal'] = 'Predicted valence and arousal between -1 and 1'
    coreml_model.author = 'Charlie Hewitt'
    coreml_model.license = 'BSD'
    coreml_model.save(path + '/AFF_NET_' + t + '_O.mlmodel')


def main(argv):
    if len(argv) == 2:
        if argv[0] == 'C' or argv[0] == 'R':
            convert(argv[0], argv[1])
            return
    raise(Exception('INPUT ERROR'))


if __name__ == '__main__':
    main(sys.argv[1:])
