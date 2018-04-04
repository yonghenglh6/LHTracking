/******************************************************************************

        Copyright (C), 2013-2018, Beijing DeepGlint Technology Limited

*******************************************************************************
File Name       : dg_common.h
Version         : Initial Draft
Author          : Jingrun Sun
Created         : 2018/01/12
Last Modified   :
Description     : All targets capture algorithm SDK common head file
Function List   :
History         :
 1.Date         : 2018/01/12
   Author       : Jingrun Sun
   Modification : Created file

 2.Date         : 2018/01/17
   Author       : Jingrun Sun
   Modification : Update definition

 3.Date         : 2018/01/19
   Author       : Jingrun Sun
   Modification : Add bool definition

 4.Date         : 2018/01/22
   Author       : Jingrun Sun
   Modification : Add frame definition

 5.Date         : 2018/02/08
   Author       : Jingrun Sun
   Modification : Add error enum(safe)
******************************************************************************/

#ifndef DG_COMMON_H
#define DG_COMMON_H

#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned char       DG_U8;
typedef char                DG_S8;
typedef unsigned int        DG_U32;
typedef int                 DG_S32;
typedef float               DG_F32;

typedef void *              DG_HANDLE;
typedef void                DG_VOID;

#define POLY_POINTS_NUM     16

#define DG_SUCCESS          0
#define DG_FAILURE          (-1)

typedef enum dgDG_BOOL
{
    DG_TRUE  = 1,
    DG_FALSE = 0
}DG_BOOL;

/****************************************************
INFO
****************************************************/

typedef enum dgDG_ERROR_E
{
    DG_ERROR_ILLEGAL_PARAM = 1,
    DG_ERROR_LOAD_MODEL,
    DG_ERROR_UNKNOWN
}DG_ERROR_E;

// Image format type
typedef enum dgDG_IMAGE_TYPE_E
{
    DG_IMAGE_YUV_SP420 = 0,
    DG_IMAGE_BGR_PACKAGE,
    DG_IMAGE_BGR_PLANAR,
    DG_IMAGE_FORMAT_BUTT
}DG_IMAGE_TYPE_E;

// Image information
typedef struct dgDG_IMAGE_S
{
    DG_IMAGE_TYPE_E         enType;
    DG_U8                   *pu8VirAddr;
    DG_U32                  u32PhyAddr;
    DG_U32                  u32Width;
    DG_U32                  u32Height;
    DG_U32                  u32Stride;
}DG_IMAGE_S;

// Frame information
typedef struct dgDG_FRAME_S
{
    DG_IMAGE_S              stImage;
    DG_U32                  u32FrameId;
}DG_FRAME_S;

// Image size
typedef struct dgDG_IMAGE_SIZE_S
{
    DG_U32                  u32Width;
    DG_U32                  u32Height;
}DG_IMAGE_SIZE_S;

// Image pixel coordinate
typedef struct dgDG_POINT_S
{
    DG_S32                  s32X;
    DG_S32                  s32Y;
}DG_POINT_S;

// Polygon definition
typedef struct dgDG_POLYGON_S
{
    DG_U32                  u32Num;
    DG_POINT_S              astPoints[POLY_POINTS_NUM];
}DG_POLYGON_S;

// Rectangle definition
typedef struct dgDG_RECT_S
{
    DG_U32                  u32X;
    DG_U32                  u32Y;
    DG_U32                  u32Width;
    DG_U32                  u32Height;
}DG_RECT_S;

#ifdef __cplusplus
}
#endif

#endif

