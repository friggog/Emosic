import sys

import coremltools


def do(t, path):
    if t == 'C':
        coreml_model = coremltools.converters.keras.convert(path,
                                                            input_names=['image'],
                                                            image_input_names=['image'],
                                                            output_names=['emotion_p'],
                                                            image_scale=1 / 255.0,
                                                            class_labels=['0', '1', '2', '3', '4', '5', '6', '7'],
                                                            predicted_feature_name='emotion')
        coreml_model.short_description = 'Predicts the emotion present in an image of a human face.'
        coreml_model.input_description['image'] = '128x128 image of human face'
        coreml_model.output_description['emotion'] = 'Predicted emotion - 1 of 8 basic emotions'
    else:
        coreml_model = coremltools.converters.keras.convert(path,
                                                            input_names=['image'],
                                                            image_input_names=['image'],
                                                            image_scale=1 / 255.0,
                                                            output_names=['valence/arousal'],
                                                            predicted_feature_name='emotion')
        coreml_model.short_description = 'Predicts the valence/arousal present in an image of a human face.'
        coreml_model.input_description['image'] = '128x128 image of human face'
        coreml_model.output_description['valence/arousal'] = 'Predicted valence and arousal between -1 and 1'
    coreml_model.author = 'Charlie Hewitt'
    coreml_model.license = 'BSD'
    coreml_model.save('MobAffNet' + t + '.mlmodel')


def main(argv):
    if len(argv) == 2:
        if argv[0] == 'C' or argv[0] == 'R':
            do(argv[0], argv[1])
            return
    raise(Exception('INPUT ERROR'))


if __name__ == '__main__':
    main(sys.argv[1:])
