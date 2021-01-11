#include <iostream>
#include "render.h"

using namespace std;

int main(int argc, char *argv[])
{
	stream_t stream_input, stream_output;
	value_t stream_data;

	cout << ">> start" << endl;

	stream_data.data = 0;
	stream_data.keep = 0xFF;
	stream_data.strb = 0xFF;
	stream_data.user = 0;
	stream_data.last = 0;
	stream_data.id = 0;
	stream_data.dest = 0;

	union_uint32_float_t u;

	float_2_to_apuint64_1(1.0, 2.0, stream_data.data); stream_input.write(stream_data);
	float_2_to_apuint64_1(3.0, 4.0, stream_data.data); stream_input.write(stream_data);
	float_2_to_apuint64_1(5.0, 6.0, stream_data.data); stream_input.write(stream_data);
	float_2_to_apuint64_1(7.0, 8.0, stream_data.data); stream_input.write(stream_data);

	float_2_to_apuint64_1(1.3, 2.3, stream_data.data); stream_input.write(stream_data);
	float_2_to_apuint64_1(3.3, 4.3, stream_data.data); stream_input.write(stream_data);
	float_2_to_apuint64_1(5.3, 6.3, stream_data.data); stream_input.write(stream_data);
	float_2_to_apuint64_1(7.3, 8.3, stream_data.data); stream_input.write(stream_data);

	float_2_to_apuint64_1(1.7, 2.7, stream_data.data); stream_input.write(stream_data);
	float_2_to_apuint64_1(3.7, 4.7, stream_data.data); stream_input.write(stream_data);
	float_2_to_apuint64_1(5.7, 6.7, stream_data.data); stream_input.write(stream_data);
	float_2_to_apuint64_1(7.7, 8.7, stream_data.data); stream_input.write(stream_data);

	float transform[3][4] = {{1.0, 0.0, 0.0, 1.0},
							 {0.0, 1.0, 0.0, 2.0},
							 {0.0, 0.0, 1.0, 3.0}};

	geometric_transform(&stream_input, &stream_output, 1, transform, 2.0);

	float fa, fb;
	apuint64_1_to_float_2(stream_output.read().data, fa, fb);
	cout << fa << " " << fb << endl;
	apuint64_1_to_float_2(stream_output.read().data, fa, fb);
	cout << fa << " " << fb << endl;
	apuint64_1_to_float_2(stream_output.read().data, fa, fb);
	cout << fa << " " << fb << endl;
	apuint64_1_to_float_2(stream_output.read().data, fa, fb);
	cout << fa << " " << fb << endl;
	apuint64_1_to_float_2(stream_output.read().data, fa, fb);
	cout << fa << " " << fb << endl;
	apuint64_1_to_float_2(stream_output.read().data, fa, fb);
	cout << fa << " " << fb << endl;
	apuint64_1_to_float_2(stream_output.read().data, fa, fb);
	cout << fa << " " << fb << endl;
	apuint64_1_to_float_2(stream_output.read().data, fa, fb);
	cout << fa << " " << fb << endl;
	apuint64_1_to_float_2(stream_output.read().data, fa, fb);
	cout << fa << " " << fb << endl;
	apuint64_1_to_float_2(stream_output.read().data, fa, fb);
	cout << fa << " " << fb << endl;
	apuint64_1_to_float_2(stream_output.read().data, fa, fb);
	cout << fa << " " << fb << endl;
	apuint64_1_to_float_2(stream_output.read().data, fa, fb);
	cout << fa << " " << fb << endl;

	cout << ">> end" << endl;

	return 0;
}
