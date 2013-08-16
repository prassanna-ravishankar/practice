#include <vtkVersion.h>
#include <vtkSmartPointer.h>
#include <vtkImageData.h>
#include <vtkContourFilter.h>
#include <vtkMarchingCubes.h>
#include <vtkVoxelModeller.h>
#include <vtkSphereSource.h>
#include <vtkPolyData.h>
#include <vtkImageCast.h>
#include <vtkXMLImageDataWriter.h>
#include <vtkActor.h>
#include <vtkPolyDataMapper.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include "vtkImageActor.h"
#include "vtkColorTransferFunction.h"
#include "vtkVolume.h"
#include "vtkVolumeProperty.h"
#include "vtkXMLImageDataReader.h"
#include "vtkPiecewiseFunction.h"
#include "vtkProperty.h"
#include "vtkSmartVolumeMapper.h"
#include "vtkFixedPointVolumeRayCastMapper.h"
#include <cstdlib>
#include <iostream>

using namespace std;

void loadRawFile(const char *filename, size_t size, unsigned char *volume)
{
  FILE *fp = fopen(filename, "rb");

  if (!fp)
    {
      fprintf(stderr, "Error opening file '%s'\n", filename);
      return;
    }

  size_t read = fread(volume, 1, size, fp);
  fclose(fp);
  printf("Read '%s', %d bytes\n", filename, read);
}

int main(int argc, char *argv[])
{

//  const char* filename = argv[1];
//  int x = atoi(argv[2]),y = atoi(argv[3]), z = atoi(argv[4]);
//  unsigned long size = x * y * z;
//  unsigned char *volume = new unsigned char[size];
//  loadRawFile(filename, size, volume);

//  vtkSmartPointer<vtkImageData> imageData = vtkSmartPointer<vtkImageData>::New();
//  imageData->SetDimensions(x,y,z);
//  imageData->SetNumberOfScalarComponents(1);
//  imageData->SetScalarTypeToUnsignedChar();

//  for(int k = 0; k < z; ++k){
//    for(int j = 0; j < y; ++j){
//      for(int i = 0; i < x; ++i){
//	int ix = i + j * y + k * y * z;
//	unsigned char* pixel = static_cast<unsigned char*>(imageData->GetScalarPointer(i,j,k));
//	pixel[0] = volume[ix];
//      }
//    }
//  }


//  //  vtkSmartPointer<vtkMarchingCubes> surface =
//  //    vtkSmartPointer<vtkMarchingCubes>::New();

//  //  surface->SetInput(imageData.GetPointer());
//  //  surface->ComputeNormalsOn();
//  //  surface->SetValue(0, 50);


//  vtkSmartPointer<vtkRenderer> renderer =
//    vtkSmartPointer<vtkRenderer>::New();
//  renderer->SetBackground(.1, .2, .3);

//  vtkSmartPointer<vtkRenderWindow> renderWindow =
//    vtkSmartPointer<vtkRenderWindow>::New();
//  renderWindow->AddRenderer(renderer);
//  vtkSmartPointer<vtkRenderWindowInteractor> interactor =
//    vtkSmartPointer<vtkRenderWindowInteractor>::New();
//  interactor->SetRenderWindow(renderWindow);

//  vtkSmartPointer<vtkVolume> vol = vtkSmartPointer<vtkVolume>::New();
//  //  vtkSmartVolumeMapper *mapper = vtkSmartVolumeMapper::New();
//  vtkSmartPointer<vtkFixedPointVolumeRayCastMapper> mapper =
//    vtkSmartPointer<vtkFixedPointVolumeRayCastMapper>::New();
//  mapper->SetInput(imageData.GetPointer());
//  vtkSmartPointer<vtkColorTransferFunction> colorFun =
//    vtkSmartPointer<vtkColorTransferFunction>::New();
//  vtkSmartPointer<vtkPiecewiseFunction> opacityFun =
//    vtkSmartPointer<vtkPiecewiseFunction>::New();

//  // Create the property and attach the transfer functions
//  vtkSmartPointer<vtkVolumeProperty> property =
//    vtkSmartPointer<vtkVolumeProperty>::New();
//  property->SetIndependentComponents(true);
//  property->SetColor( colorFun );
//  property->SetScalarOpacity( opacityFun );
//  property->SetInterpolationTypeToLinear();
//  vol->SetProperty( property );
//  vol->SetMapper( mapper );
//  colorFun->AddRGBSegment(0.0, 1.0, 1.0, 1.0, 255.0, 1.0, 1.0, 1.0 );
//  double opacityLevel = 2048/32;
//  double opacityWindow = 4096/32;
//  opacityFun->AddSegment( opacityLevel - 0.5*opacityWindow, 0.0,
//                          opacityLevel + 0.5*opacityWindow, 1.0 );
//  mapper->SetBlendModeToMaximumIntensity();

//  vtkSmartPointer<vtkPolyDataMapper> mapper =
//    vtkSmartPointer<vtkPolyDataMapper>::New();
//  //  mapper->SetInputConnection(surface->GetOutputPort());

//  //  vtkSmartPointer<vtkImageActor> actor =
//  //    vtkSmartPointer<vtkImageActor>::New();
//  // // actor->SetMapper(mapper);

//  //  actor->SetInput(imageData);
//  //  renderer->AddActor(actor);

//  renderWindow->SetSize(600,600);
//  renderWindow->Render();
//  renderer->AddVolume(vol);
//  renderer->ResetCamera();
//  renderWindow->Render();
//  interactor->Start();

//  delete volume;
//  return EXIT_SUCCESS;
}
