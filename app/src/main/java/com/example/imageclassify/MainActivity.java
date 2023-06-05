package com.example.imageclassify;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.example.imageclassify.ml.LiteModelImagenetMobilenetV3Large100224Classification5Metadata1;

import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.label.Category;

import java.io.IOException;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    Button selectButton, captureButton;
    ImageView imageView;
    TextView resultTv;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        selectButton = findViewById(R.id.btnSelectImage);
        captureButton = findViewById(R.id.btnCaptureImage);
        imageView = findViewById(R.id.iV1);
        resultTv = findViewById(R.id.tvResult);

        if (ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[] {Manifest.permission.CAMERA}, 1);
        }

        selectButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
                intent.setType("image/*");
                startActivityForResult(intent, 10);
            }
        });
        captureButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(intent, 12);
            }
        });
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == 1) {
            if (grantResults[0] != PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(getApplicationContext(), "카메라 권환 필요함...", Toast.LENGTH_SHORT).show();
                captureButton.setEnabled(false);
            }
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        Bitmap bitmap = null;

        if (data != null) {
            if (requestCode == 10) {
                Uri uri = data.getData();

                try {
                    bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
            else if (requestCode == 12) {
                bitmap = (Bitmap) data.getExtras().get("data");
            }
        }

        if (bitmap != null) {
            imageView.setImageBitmap(bitmap);
            imageClassify(bitmap);
        }
        super.onActivityResult(requestCode, resultCode, data);
    }

    private void imageClassify(Bitmap bitmap) {
        try {
            LiteModelImagenetMobilenetV3Large100224Classification5Metadata1 model = LiteModelImagenetMobilenetV3Large100224Classification5Metadata1.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorImage image = TensorImage.fromBitmap(bitmap);

            // Runs model inference and gets result.
            LiteModelImagenetMobilenetV3Large100224Classification5Metadata1.Outputs outputs = model.process(image);
            List<Category> logit = outputs.getLogitAsCategoryList();

            Category category= logit.get(0);
            float maxScore = category.getScore();
            int maxIndex = 0;

            for (int i=1; i<logit.size(); i++) {
                category = logit.get(i);
                if(category.getScore() > maxScore) {
                    maxScore = category.getScore();
                    maxIndex = i;
                }
            }

            resultTv.setText(logit.get(maxIndex).getLabel() + "\n확률점수: " + maxScore);

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
            e.printStackTrace();
        }


    }
}