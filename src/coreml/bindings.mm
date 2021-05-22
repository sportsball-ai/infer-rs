#import <CoreML/CoreML.h>
#import <Metal/Metal.h>
#include <cstdint>

#if !__has_feature(objc_arc)
#error "ARC is off"
#endif

extern "C" const void* open_coreml_model(const char* path, uint32_t counter) {
    @autoreleasepool {
        NSString* pathString = [NSString stringWithUTF8String:path];
        NSURL* pathURL = [NSURL fileURLWithPath:pathString];

        NSError *error = nil;

        NSURL* compiledURL = [MLModel compileModelAtURL:pathURL error:&error];
        if (error != nil) {
            NSLog(@"%@", error);
            return NULL;
        }

        MLModelConfiguration* config = [MLModelConfiguration new];

        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        if (devices.count > 1) {
            config.preferredMetalDevice = devices[counter % devices.count];
        }

        MLModel* model = [MLModel modelWithContentsOfURL:compiledURL configuration:config error:&error];
        [[NSFileManager defaultManager] removeItemAtURL:compiledURL error:NULL];
        if (error != nil) {
            NSLog(@"%@", error);
            return NULL;
        }

        return (__bridge_retained void*)model;
    }
}

extern "C" uint32_t mlmultiarray_dimensionality(const void* multiarray) {
    @autoreleasepool {
        return ((__bridge MLMultiArray*)multiarray).shape.count;
    }
}

extern "C" void mlmultiarray_get_shape(const void* multiarray, uint32_t* dest) {
    @autoreleasepool {
        for (NSNumber* n in ((__bridge MLMultiArray*)multiarray).shape) {
            *dest = n.unsignedIntValue;
            ++dest;
        }
    }
}

extern "C" const void* mlmultiarray_data(const void* multiarray) {
    @autoreleasepool {
        return ((__bridge MLMultiArray*)multiarray).dataPointer;
    }
}

extern "C" uint32_t mlmodel_output_count(const void* model) {
    @autoreleasepool {
        return ((__bridge MLModel*)model).modelDescription.outputDescriptionsByName.count;
    }
}

extern "C" void mlmodel_get_output_names(const void* model, const void** dest) {
    @autoreleasepool {
        for (id name in ((__bridge MLModel*)model).modelDescription.outputDescriptionsByName) {
            *dest = (__bridge_retained void*)name;
            ++dest;
        }
    }
}

extern "C" const char* nsstring_utf8(const void* s) {
    @autoreleasepool {
        return [((__bridge NSString*)s) UTF8String];
    }
}

extern "C" const void* mlfeatureprovider_multiarray_by_name(const void* provider, const char* name) {
    @autoreleasepool {
        MLFeatureValue* value = [(__bridge id)provider featureValueForName:[NSString stringWithUTF8String:name]];
        if (!value) {
            return NULL;
        }
        return (__bridge_retained void*)value.multiArrayValue;
    }
}

extern "C" const void* mlmodel_predict(
    const void* model,
    const char** inputNames,
    const void** inputDataPtrs,
    const uint32_t* inputDimensionalities,
    const uint32_t** inputShapes,
    uint32_t inputCount
) {
    @autoreleasepool {
        NSError* error = nil;

        NSMutableDictionary* inputs = [NSMutableDictionary dictionaryWithCapacity:inputCount];

        for (uint32_t i = 0; i < inputCount; ++i) {
            uint32_t dimensionality = inputDimensionalities[i];

            NSMutableArray* shape = [NSMutableArray arrayWithCapacity:dimensionality];
            NSMutableArray* reversedStrides = [NSMutableArray arrayWithCapacity:dimensionality];

            uint32_t p = 1;
            for (uint32_t d = 0; d < dimensionality; ++d) {
                [shape insertObject:[NSNumber numberWithUnsignedInt:inputShapes[i][d]] atIndex:d];
                [reversedStrides insertObject:[NSNumber numberWithUnsignedInt:p] atIndex:d];
                p *= inputShapes[i][dimensionality - d - 1];
            }

            MLMultiArray* array = [[MLMultiArray alloc] initWithDataPointer:(void*)inputDataPtrs[i]
                shape:shape
                dataType:MLMultiArrayDataTypeFloat32
                strides:[[reversedStrides reverseObjectEnumerator] allObjects]
                deallocator:nil
                error:&error
            ];
            if (error != nil) {
                NSLog(@"%@", error);
                return NULL;
            }

            inputs[[NSString stringWithUTF8String:inputNames[i]]] = [MLFeatureValue featureValueWithMultiArray:array];
        }

        MLDictionaryFeatureProvider* inputProvider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:inputs error:&error];
        if (error != nil) {
            NSLog(@"%@", error);
            return NULL;
        }

        id<MLFeatureProvider> outputProvider = [(__bridge id)model predictionFromFeatures:inputProvider error:&error];
        if (error != nil) {
            NSLog(@"%@", error);
            return NULL;
        }

        return (__bridge_retained void*)outputProvider;
    }
}

extern "C" void release_object(const void* obj) {
    @autoreleasepool {
        id m = (__bridge_transfer id)obj;
        m = nil;
    }
}
