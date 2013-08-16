//#include <itkImageFileReader.h>
#include <itkNiftiImageIO.h>
#include <itkImageFileReader.h>
#include <itkImageToVTKImageFilter.h>
#include <vtkVersion.h>
#include <vtkSmartPointer.h>
#include <vtkGPUVolumeRayCastMapper.h>
#include <vtkColorTransferFunction.h>
#include <vtkPiecewiseFunction.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkVolumeProperty.h>
#include <vtkMatrix4x4.h>
#include <vtkAxesActor.h>


int main(int argc, char *argv[])
{

// ITK: Read the image
// ------------------------------------------------------------------------------
//    typedef itk::Image<unsigned char, 3> VisualizingImageType;
//    typedef itk::NiftiImageIO  ReaderType;
//    ReaderType::Pointer nifti = ReaderType::New();
////    ReaderType::Pointer reader = ReaderType::New();
//    nifti->SetFileName(argv[1]);

//    reader->Update();
//    VisualizingImageType::Pointer image=reader->GetOutput();


    typedef itk::Image<unsigned char, 3> VisualizingImageType;
       typedef itk::ImageFileReader< VisualizingImageType >  ReaderType;
       ReaderType::Pointer reader = ReaderType::New();
       reader->SetFileName( argv[1] );
       reader->Update();
       VisualizingImageType::Pointer image=reader->GetOutput();

// VTK: Create window and renderer
// ------------------------------------------------------------------------------
    vtkSmartPointer<vtkRenderWindow> renWin =
            vtkSmartPointer<vtkRenderWindow>::New();
    vtkSmartPointer<vtkRenderer> ren1 = vtkSmartPointer<vtkRenderer>::New();
    ren1->SetBackground(0.5f,0.5f,1.0f);
    renWin->AddRenderer(ren1);
    renWin->SetSize(1280,1024);
    vtkSmartPointer<vtkRenderWindowInteractor> iren =
            vtkSmartPointer<vtkRenderWindowInteractor>::New();
    iren->SetRenderWindow(renWin);
    renWin->Render(); // make sure we have an OpenGL context.


// ITK: ITK to VTK image conversion
// ------------------------------------------------------------------------------
    typedef itk::ImageToVTKImageFilter<VisualizingImageType> itkVtkConverter;
    itkVtkConverter::Pointer conv=itkVtkConverter::New();
    conv->SetInput(image);


// VTK: Creating a volume mapper with the input image
// ------------------------------------------------------------------------------
    vtkSmartPointer<vtkGPUVolumeRayCastMapper> volumeMapper =
            vtkSmartPointer<vtkGPUVolumeRayCastMapper>::New();
#if VTK_MAJOR_VERSION <= 5
    volumeMapper->SetInput(conv->GetOutput());
#else
    conv->Update();
    volumeMapper->SetInputData(conv->GetOutput());
#endif


// VTK: Setting volume visualization properties (transparency and color mapping)
// ------------------------------------------------------------------------------
    vtkSmartPointer<vtkVolumeProperty> volumeProperty =
            vtkSmartPointer<vtkVolumeProperty>::New();

    vtkSmartPointer<vtkPiecewiseFunction> compositeOpacity =
            vtkSmartPointer<vtkPiecewiseFunction>::New();
    compositeOpacity->AddPoint(0.0,    0.0); 	// This is a custom setting /!\
    compositeOpacity->AddPoint(1000.0, 1.0); 	// for my test volume!
    volumeProperty->SetScalarOpacity(compositeOpacity);
    volumeProperty->SetScalarOpacityUnitDistance(1.0);

    vtkSmartPointer<vtkColorTransferFunction> color =
            vtkSmartPointer<vtkColorTransferFunction>::New();
    color->AddRGBPoint(0.0,    0.0,0.0,0.0); 	// This is a custom setting /!\
    color->AddRGBPoint(1000.0, 1.0,1.0,1.0); 	// for my test volume!
    volumeProperty->SetColor(color);

    vtkSmartPointer<vtkVolume> volume =
            vtkSmartPointer<vtkVolume>::New();
    volume->SetMapper(volumeMapper);
    volume->SetProperty(volumeProperty);


// VTK: Positioning the volume where it should be...
// I don't know if this is valid in all scenarios :S
// ------------------------------------------------------------------------------
    // Here we take care of position and orientation
    // so that volume is in DICOM patient physical space
    VisualizingImageType::DirectionType d=image->GetDirection();
    vtkMatrix4x4 *mat=vtkMatrix4x4::New();
    //start with identity matrix
    for (int i=0; i<3; i++)
        for (int k=0; k<3; k++)
            mat->SetElement(i,k, d(i,k));

    // Counteract the built-in translation by origin
    VisualizingImageType::PointType origin=image->GetOrigin();
    volume->SetPosition(-origin[0], -origin[1], -origin[2]);
    // Add translation to the user matrix
    for (int i=0; i<3; i++)
    {
        mat->SetElement(i,3, origin[i]);
    }
    volume->SetUserMatrix(mat);


// VTK: Adding reference axes
// ------------------------------------------------------------------------------
    // Add coordinate system axes, so we have a reference for position and orientation
    vtkSmartPointer<vtkAxesActor> axes = vtkSmartPointer<vtkAxesActor>::New();
    axes->SetTotalLength(250,250,250);
    axes->SetShaftTypeToCylinder();
    axes->SetCylinderRadius(0.01);
    ren1->AddActor(axes);


// VTK: Ultimating details ;)
// ------------------------------------------------------------------------------
    ren1->AddVolume( volume );
    ren1->ResetCamera();

    renWin->Render();
    renWin->Render();
    renWin->Render();

    iren->Start();

    return EXIT_SUCCESS;
}
