![](https://img.shields.io/badge/language-swift-blue.svg)
![](https://img.shields.io/badge/version-4.0-red.svg)
### Quick Start:
let classifier = YoloTinyV1Classifier()

classifier.loadModel()

classifier.classifyImage(your-image)

let boxes = classifier.result()
### Installation:
#### • CocoaPods

```
use_frameworks!
pod 'YoloIOS', '~>1.0.0'


post_install do |installer|
    installer.pods_project.targets.each do |target|
        target.build_configurations.each do |config|
            if ['YoloIOS'].include? target.name
             config.build_settings['SWIFT_VERSION'] = '4.0'
            end
        end
    end
end
```


### Technical details:
- Swift 4.0, Object C++
- Tensorflow
- OpenCV 3.2.0
### Licenses
All source code is licensed under the MIT License.

If you use it, i'll be happy to know about it.
